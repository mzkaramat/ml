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
package com.cloudera.science.ml.classifier.simple;

import com.cloudera.science.ml.classifier.core.Classifier;
import com.cloudera.science.ml.classifier.core.LinearClassifier;
import com.cloudera.science.ml.classifier.core.OnlineLearnerParams;
import com.cloudera.science.ml.classifier.core.WeightVector;
import com.cloudera.science.ml.core.vectors.LabeledVector;

/**
 *
 */
public class LogRegOnlineLearner implements SimpleOnlineLearner {

  private WeightVector weights;
  private LinearClassifier classifier;
  private final OnlineLearnerParams params;
  private int iteration;
  
  public LogRegOnlineLearner(OnlineLearnerParams params) {
    this.params = params;
    this.iteration = 0;
  }

  @Override
  public Classifier getClassifier() {
    return classifier;
  }
  
  @Override
  public OnlineLearnerParams getParams() {
    return params;
  }

  @Override
  public boolean update(LabeledVector x) {
    if (weights == null) {
      weights = new WeightVector(x.size());
      classifier = new LinearClassifier(weights);
    }
    iteration++;
    double label = x.getLabel();
    if (label == 0.0) {
      label = -1.0;
    }
    double eta = params.eta(iteration);
    double loss = label / (1 + Math.exp(label * weights.innerProduct(x)));
    weights.regularizeL2(eta, params.lambda());    
    weights.addVector(x, (eta * loss));
    params.updateWeights(weights, iteration);
    return true;
  }
}
