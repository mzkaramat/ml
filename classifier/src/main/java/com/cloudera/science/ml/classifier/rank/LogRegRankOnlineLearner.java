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
package com.cloudera.science.ml.classifier.rank;

import com.cloudera.science.ml.classifier.core.Classifier;
import com.cloudera.science.ml.classifier.core.LinearClassifier;
import com.cloudera.science.ml.classifier.core.LogRegClassifier;
import com.cloudera.science.ml.classifier.core.OnlineLearnerParams;
import com.cloudera.science.ml.classifier.core.WeightVector;
import com.cloudera.science.ml.core.vectors.LabeledVector;

/**
 *
 */
public class LogRegRankOnlineLearner implements RankOnlineLearner {

  private WeightVector weights;
  private final LinearClassifier classifer;
  private final OnlineLearnerParams params;
  private int iteration;
  
  public LogRegRankOnlineLearner(OnlineLearnerParams params) {
    this.classifer = new LinearClassifier(weights);
    this.params = params;
    this.iteration = 0;
  }

  @Override
  public Classifier getClassifier() {
    return classifer;
  }
  
  @Override
  public OnlineLearnerParams getParams() {
    return params;
  }

  @Override
  public boolean update(LabeledVector a, LabeledVector b) {
    if (weights == null) {
      weights = new WeightVector(a.size());
    }
    iteration++;
    double eta = params.eta(iteration);
    double y = (a.getLabel() > b.getLabel()) ? 1.0 :
        (a.getLabel() < b.getLabel()) ? -1.0 : 0.0;
    double loss = y / (1.0 + Math.exp(y * weights.innerProductOnDifference(a, b)));
    weights.regularizeL2(eta, params.lambda());    

    weights.addVector(a, (eta * loss));
    weights.addVector(b, (-1.0 * eta * loss));
    
    params.updateWeights(weights, iteration);
    return true;
  }
}
