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
package com.cloudera.science.ml.classifier.core;

import org.apache.mahout.math.Vector;



public class LogRegClassifier implements Classifier {
  private WeightVector weights;
  
  public LogRegClassifier(WeightVector weights) {
    this.weights = weights;
  }
  
  @Override
  public Double apply(Vector features) {
    double expOdds = Math.exp(-weights.innerProduct(features));
    return expOdds / (1.0 + expOdds);
  }
  
  @Override
  public WeightVector getWeights() {
    return weights;
  }
  
  @Override
  public String toString() {
    return "LogRegClassifier(" + weights.toString() + ")";
  }
}