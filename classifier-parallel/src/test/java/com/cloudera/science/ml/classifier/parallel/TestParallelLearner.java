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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;

import static junit.framework.Assert.assertEquals;

import org.apache.crunch.PCollection;
import org.apache.crunch.Pair;
import org.apache.crunch.impl.mem.MemPipeline;
import org.apache.crunch.materialize.pobject.CollectionPObject;
import org.junit.Test;

import com.cloudera.science.ml.classifier.core.EtaUpdate;
import com.cloudera.science.ml.classifier.core.OnlineLearnerParams;
import com.cloudera.science.ml.classifier.core.OnlineLearnerRun;
import com.cloudera.science.ml.classifier.parallel.FitFn;
import com.cloudera.science.ml.classifier.parallel.ParallelLearner;
import com.cloudera.science.ml.classifier.parallel.SimpleFitFn;
import com.cloudera.science.ml.classifier.simple.LogRegOnlineLearner;
import com.cloudera.science.ml.classifier.simple.SimpleOnlineLearner;
import com.cloudera.science.ml.core.vectors.LabeledVector;
import com.cloudera.science.ml.core.vectors.LabeledVectors;
import com.cloudera.science.ml.parallel.crossfold.CrossfoldFn;
import com.cloudera.science.ml.parallel.distribute.DistributeFn;
import com.cloudera.science.ml.parallel.distribute.SimpleDistributeFn;
import com.cloudera.science.ml.parallel.fn.ShuffleFn;
import com.cloudera.science.ml.parallel.types.MLAvros;

public class TestParallelLearner {
  private static final long seed = 1729L;
  
  private static final double YES = 1.0;
  private static final double NO = 0.0;
  private static final PCollection<LabeledVector> TRAINING_DATA = 
      MemPipeline.typedCollectionOf(
          MLAvros.labeledVector(),
          LabeledVectors.of(NO, -5.0, -10.0),
          LabeledVectors.of(NO, -10.0, -5.0),
          LabeledVectors.of(YES, 5.0, 10.0),
          LabeledVectors.of(YES, 10.0, 5.0),
          LabeledVectors.of(NO, -5.0, -10.0),
          LabeledVectors.of(NO, -10.0, -5.0),
          LabeledVectors.of(YES, 5.0, 10.0),
          LabeledVectors.of(YES, 10.0, 5.0),
          LabeledVectors.of(NO, -5.0, -10.0),
          LabeledVectors.of(NO, -10.0, -5.0),
          LabeledVectors.of(YES, 5.0, 10.0),
          LabeledVectors.of(YES, 10.0, 5.0));
  
  @Test
  public void testParallelLearner() {
    final int numFolds = 3;
    final int numPortions = 2;
    
    OnlineLearnerParams.Builder builder = OnlineLearnerParams.builder();
    builder.pegasos(false).etaUpdate(EtaUpdate.BASIC_ETA);
    List<SimpleOnlineLearner> learners = new ArrayList<SimpleOnlineLearner>();
    OnlineLearnerParams params1 = builder.L2(0.0).build();
    OnlineLearnerParams params2 = builder.L2(1.0).build();
    learners.add(new LogRegOnlineLearner(params1));
    learners.add(new LogRegOnlineLearner(params2));
    
    ShuffleFn<LabeledVector> shuffleFn = new ShuffleFn<LabeledVector>(seed);
    CrossfoldFn<Pair<Integer, LabeledVector>> crossfoldFn =
        new CrossfoldFn<Pair<Integer, LabeledVector>>(numFolds, seed);
    DistributeFn<Integer, Pair<Integer, LabeledVector>> distributeFn = 
        new SimpleDistributeFn<Integer, Pair<Integer, LabeledVector>>(numPortions, seed);
    FitFn fitFn = new SimpleFitFn(learners);

    ParallelLearner learner = new ParallelLearner();
    PCollection<OnlineLearnerRun> pruns = learner.runPipeline(TRAINING_DATA, shuffleFn,
        crossfoldFn, distributeFn, fitFn);
    
    Collection<OnlineLearnerRun> runs =
        new CollectionPObject<OnlineLearnerRun>(pruns).getValue();
    
    assertEquals(2 * 3 * 2, runs.size());
    List<OnlineLearnerRun> expected = Arrays.asList(
        new OnlineLearnerRun(null, null, params1, 0, 0, 0),
        new OnlineLearnerRun(null, null, params1, 0, 1, 0),
        new OnlineLearnerRun(null, null, params1, 1, 0, 0),
        new OnlineLearnerRun(null, null, params1, 1, 1, 0),
        new OnlineLearnerRun(null, null, params1, 2, 0, 0),
        new OnlineLearnerRun(null, null, params1, 2, 1, 0),
        new OnlineLearnerRun(null, null, params2, 0, 0, 1),
        new OnlineLearnerRun(null, null, params2, 0, 1, 1),
        new OnlineLearnerRun(null, null, params2, 1, 0, 1),
        new OnlineLearnerRun(null, null, params2, 1, 1, 1),
        new OnlineLearnerRun(null, null, params2, 2, 0, 1),
        new OnlineLearnerRun(null, null, params2, 2, 1, 1));
    
    compareRuns(expected, runs);
  }
  
  @Test
  public void testDeterminism() {
    final int numFolds = 3;
    final int numPortions = 2;
        
    OnlineLearnerParams.Builder builder = OnlineLearnerParams.builder();
    builder.pegasos(false).etaUpdate(EtaUpdate.BASIC_ETA);
    List<SimpleOnlineLearner> learners = new ArrayList<SimpleOnlineLearner>();
    OnlineLearnerParams params1 = builder.L2(0.0).build();
    OnlineLearnerParams params2 = builder.L2(1.0).build();
    learners.add(new LogRegOnlineLearner(params1));
    learners.add(new LogRegOnlineLearner(params2));
    
    ShuffleFn<LabeledVector> shuffleFn = new ShuffleFn<LabeledVector>(seed);
    CrossfoldFn<Pair<Integer, LabeledVector>> crossfoldFn =
        new CrossfoldFn<Pair<Integer, LabeledVector>>(numFolds, seed);
    DistributeFn<Integer, Pair<Integer, LabeledVector>> distributeFn = 
        new SimpleDistributeFn<Integer, Pair<Integer, LabeledVector>>(numPortions, seed);
    FitFn fitFn = new SimpleFitFn(learners);
    
    ParallelLearner learner = new ParallelLearner();
    
    PCollection<OnlineLearnerRun> pruns1 = learner.runPipeline(TRAINING_DATA, shuffleFn,
        crossfoldFn, distributeFn, fitFn);
    PCollection<OnlineLearnerRun> pruns2 = learner.runPipeline(TRAINING_DATA, shuffleFn,
        crossfoldFn, distributeFn, fitFn);
    
    Iterator<OnlineLearnerRun> iter1 =
        new CollectionPObject<OnlineLearnerRun>(pruns1).getValue().iterator();
    Iterator<OnlineLearnerRun> iter2 =
        new CollectionPObject<OnlineLearnerRun>(pruns2).getValue().iterator();
    
    while (iter1.hasNext()) {
      OnlineLearnerRun run1 = iter1.next();
      OnlineLearnerRun run2 = iter2.next();
      assertEquals(run1.getFold(), run2.getFold());
      assertEquals(run1.getPartition(), run2.getPartition());
      assertEquals(run1.getParams().lambda(), run2.getParams().lambda());
      assertEquals(run1.getClassifier().getWeights(),
          run2.getClassifier().getWeights());
    }
  }
  
  private void compareRuns(Collection<OnlineLearnerRun> expected,
      Collection<OnlineLearnerRun> actual) {
    RunComparator comparator = new RunComparator();
    List<OnlineLearnerRun> expectedList = new ArrayList<OnlineLearnerRun>();
    expectedList.addAll(expected);
    Collections.sort(expectedList, comparator);
    List<OnlineLearnerRun> actualList = new ArrayList<OnlineLearnerRun>();
    actualList.addAll(actual);
    Collections.sort(actualList, comparator);
    
    for (int i = 0; i < expectedList.size(); i++) {
      OnlineLearnerRun expectedRun = expectedList.get(i);
      OnlineLearnerRun actualRun = actualList.get(i);
      assertEquals(expectedRun.getFold(), actualRun.getFold());
      assertEquals(expectedRun.getPartition(), actualRun.getPartition());
      assertEquals(expectedRun.getParams().lambda(),
          actualRun.getParams().lambda(), .0001);
    }
  }
  
  private class RunComparator implements Comparator<OnlineLearnerRun> {
    @Override
    public int compare(OnlineLearnerRun o1, OnlineLearnerRun o2) {
      int ret = o1.getFold() - o2.getFold();
      if (ret != 0) {
        return ret;
      }
      ret = o1.getPartition() - o2.getPartition();
      if (ret != 0) {
        return ret;
      }
      return (int)Math.signum(o1.getParams().lambda() - o2.getParams().lambda());
    }
  }
}
