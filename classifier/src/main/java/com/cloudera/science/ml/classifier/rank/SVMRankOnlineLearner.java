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
import com.cloudera.science.ml.classifier.core.OnlineLearnerParams;
import com.cloudera.science.ml.classifier.core.WeightVector;
import com.cloudera.science.ml.core.vectors.LabeledVector;

/**
 *
 */
public class SVMRankOnlineLearner implements RankOnlineLearner {

  private WeightVector weights;
  private LinearClassifier classifier;
  private final OnlineLearnerParams params;
  private int iteration;
  
  public SVMRankOnlineLearner(OnlineLearnerParams params) {
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
  public boolean update(LabeledVector a, LabeledVector b) {
    if (weights == null) {
      weights = new WeightVector(a.size());
      classifier = new LinearClassifier(weights);
    }
    iteration++;
    double eta = params.eta(iteration);
    double y = (a.getLabel() > b.getLabel()) ? 1.0 :
        (a.getLabel() < b.getLabel()) ? -1.0 : 0.0;
    double p = y * weights.innerProductOnDifference(a, b);

    weights.regularizeL2(eta, params.lambda());

    // If (a - b) has non-zero loss, perform gradient step.         
    if (p < 1.0 && y != 0.0) {
      weights.addVector(a, (eta * y));
      weights.addVector(b, (-1.0 * eta * y));
    }

    params.updateWeights(weights, iteration);
    return (p < 1.0 && y != 0.0);
  }
}
