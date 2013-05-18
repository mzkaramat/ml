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
import com.cloudera.science.ml.classifier.core.OnlineLearner;
import com.cloudera.science.ml.classifier.core.LogRegClassifier;
import com.cloudera.science.ml.classifier.core.WeightVector;
import com.cloudera.science.ml.core.vectors.LabeledVector;

/**
 *
 */
public class LogRegOnlineLearner implements SimpleOnlineLearner {

  private final WeightVector weights;
  private final LogRegClassifier classifier;
  private final OnlineLearner.Params params;
  private int iteration;
  
  public LogRegOnlineLearner(OnlineLearner.Params params) {
    this.weights = params.createWeights();
    this.classifier = new LogRegClassifier(weights);
    this.params = params;
    this.iteration = 0;
  }

  @Override
  public Classifier getClassifier() {
    return classifier;
  }

  @Override
  public boolean update(LabeledVector x) {
    iteration++;
    double eta = params.eta(iteration);
    double loss = x.getLabel() / (1 + Math.exp(x.getLabel() * weights.innerProduct(x)));
    weights.regularizeL2(eta, params.lambda());    
    weights.addVector(x, (eta * loss));
    params.updateWeights(weights, iteration);
    return true;
  }
}
