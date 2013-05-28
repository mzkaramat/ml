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

import org.apache.crunch.DoFn;
import org.apache.crunch.Emitter;
import org.apache.crunch.Pair;
import org.apache.crunch.Tuple3;

import com.cloudera.science.ml.classifier.core.Classifier;
import com.cloudera.science.ml.classifier.core.ModelScore;
import com.cloudera.science.ml.classifier.core.OnlineLearnerRun;
import com.cloudera.science.ml.core.vectors.LabeledVector;
import com.google.common.collect.Multimap;

public class ClassifyFn extends DoFn<Pair<Integer, LabeledVector>, Pair<Tuple3<Integer, Integer, Integer>, ModelScore>> {
  
  private Multimap<Integer, OnlineLearnerRun> runsByFold;
  
  public ClassifyFn(Multimap<Integer, OnlineLearnerRun> runsByFold) {
    this.runsByFold = runsByFold;
  }
  
  @Override
  public void process(Pair<Integer, LabeledVector> vec, Emitter<Pair<Tuple3<Integer, Integer, Integer>, ModelScore>> emitter) {
    for (OnlineLearnerRun run : runsByFold.get(vec.first())) {
      Classifier classifier = run.getClassifier();
      Double result = classifier.apply(vec.second());
      if (Double.isNaN(result)) {
        getCounter("ML Counters", "Result NaNs").increment(1);
        return;
      }
      double label = vec.second().getLabel();
      if (label == Double.NaN) {
        getCounter("ML Counters", "Label NaNs").increment(1);
        return;
      }
      
      int falsePositive = (label == 0.0 && result == 1.0) ? 1 : 0;
      int truePositive = (label == 1.0 && result == 1.0) ? 1 : 0;
      int falseNegative = (label == 1.0 && result == 0.0) ? 1 : 0;
      int trueNegative = (label == 0.0 && result == 0.0) ? 1 : 0;
      
      emitter.emit(Pair.of(Tuple3.of(run.getFold(), run.getPartition(), run.getParamsVersion()),
          new ModelScore(trueNegative, falseNegative, truePositive, falsePositive)));
    }
  }

}
