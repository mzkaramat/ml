/**
 * Copyright (c) 2013, Cloudera, Inc. All Rights Reserved.
 *
 * Cloudera, Inc. licenses this file to you under the Apache License,
 * Version 2.0 (the "License"). You may not use this file except in
 * compliance with the License. You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * This software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
 * CONDITIONS OF ANY KIND, either express or implied. See the License for
 * the specific language governing permissions and limitations under the
 * License.
 */
package com.cloudera.science.ml.classifier.parallel;

import java.io.Serializable;
import java.util.Collection;
import java.util.Map;

import org.apache.crunch.DoFn;
import org.apache.crunch.Emitter;
import org.apache.crunch.PCollection;
import org.apache.crunch.PGroupedTable;
import org.apache.crunch.PTable;
import org.apache.crunch.Pair;
import org.apache.crunch.Tuple3;
import org.apache.crunch.Tuple4;
import org.apache.crunch.materialize.pobject.CollectionPObject;
import org.apache.crunch.types.PTypeFamily;

import com.cloudera.science.ml.classifier.core.ModelScore;
import com.cloudera.science.ml.classifier.core.OnlineLearnerRun;
import com.cloudera.science.ml.classifier.core.OnlineLearnerRuns;
import com.cloudera.science.ml.classifier.parallel.types.ClassifierAvros;
import com.cloudera.science.ml.core.vectors.LabeledVector;
import com.cloudera.science.ml.parallel.crossfold.Crossfold;
import com.google.common.collect.HashMultimap;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Multimap;

/**
 * Computes scores for a set of models trained by {@link ParallelLearner}
 */
public class ParallelEvaluator implements Serializable {
  /**
   * Score the models we have trained
   * @return a collection of models annotated with their score
   */
  public Collection<Pair<OnlineLearnerRun, ModelScore>> evaluateModels(
      OnlineLearnerRuns trainOutput, PCollection<LabeledVector> training) {
    Crossfold crossfold = new Crossfold(trainOutput.getNumFolds(), trainOutput.getSeed());
    PCollection<Pair<Integer, LabeledVector>> validateSets = crossfold.apply(training);
    Collection<OnlineLearnerRun> runs = trainOutput.getRuns();
    
    PTypeFamily ptf = validateSets.getTypeFamily();

    // Classify all the training data according to the models
    Multimap<Integer, OnlineLearnerRun> runsByFold = indexRunsByFold(runs);
    ClassifyFn classifyFn = new ClassifyFn(runsByFold);
    PTable<Tuple3<Integer, Integer, Integer>, ModelScore> results = validateSets.parallelDo(classifyFn,
            ptf.tableOf(ptf.triples(ptf.ints(), ptf.ints(), ptf.ints()), ClassifierAvros.modelScore()));
    
    PGroupedTable<Tuple3<Integer, Integer, Integer>, ModelScore> grouped = results.groupByKey();
    
    // Aggregate the scores on the reduce side
    PCollection<Tuple4<Integer, Integer, Integer, ModelScore>> pmodelScores = grouped.parallelDo(
        new DoFn<Pair<Tuple3<Integer, Integer, Integer>, Iterable<ModelScore>>, Tuple4<Integer, Integer, Integer, ModelScore>>() {

      @Override
      public void process(Pair<Tuple3<Integer, Integer, Integer>, Iterable<ModelScore>> modelResults,
          Emitter<Tuple4<Integer, Integer, Integer, ModelScore>> emitter) {
        ModelScore total = new ModelScore(0, 0, 0, 0);
        for (ModelScore score : modelResults.second()) {
          total.merge(score);
        }
        
        emitter.emit(Tuple4.of(modelResults.first().first(),
            modelResults.first().second(), modelResults.first().third(), total));
      }
    }, ptf.quads(ptf.ints(), ptf.ints(), ptf.ints(), ClassifierAvros.modelScore()));
    
    Collection<Tuple4<Integer, Integer, Integer, ModelScore>> modelScores =
        new CollectionPObject<Tuple4<Integer, Integer, Integer, ModelScore>>(pmodelScores).getValue();
    
    return joinRunsWithScores(modelScores, runs);
  }
  
  private Collection<Pair<OnlineLearnerRun, ModelScore>> joinRunsWithScores(
      Collection<Tuple4<Integer, Integer, Integer, ModelScore>> modelScores,
      Collection<OnlineLearnerRun> runs) {
    Map<Tuple3<Integer, Integer, Integer>, ModelScore> map =
        Maps.newHashMap();
    for (Tuple4<Integer, Integer, Integer, ModelScore> modelScore : modelScores) {
      map.put(Tuple3.of(modelScore.first(), modelScore.second(), modelScore.third()),
          modelScore.fourth());
    }
    Collection<Pair<OnlineLearnerRun, ModelScore>> runsWithScores =
        Lists.newArrayList();
    for (OnlineLearnerRun run : runs) {
      runsWithScores.add(
          Pair.of(run, map.get(Tuple3.of(run.getFold(), run.getPartition(), run.getParamsVersion()))));
    }
    return runsWithScores;
  }
  
  private Multimap<Integer, OnlineLearnerRun> indexRunsByFold(
      Collection<OnlineLearnerRun> runs) {
    Multimap<Integer, OnlineLearnerRun> runsByFold = HashMultimap.create();
    for (OnlineLearnerRun run : runs) {
      runsByFold.put(run.getFold(), run);
    }
    return runsByFold;
  }
}
