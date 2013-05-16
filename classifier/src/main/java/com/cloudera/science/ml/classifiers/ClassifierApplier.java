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

package com.cloudera.science.ml.classifiers;

import java.util.ArrayList;
import java.util.List;

import com.cloudera.science.ml.core.vectors.LabeledVector;

public class ClassifierApplier {
  float singleSvmPrediction(LabeledVector x,
      MutableVector w) {
    return w.innerProduct(x);
  }

  float singleLogisticPrediction(LabeledVector x,
      MutableVector w) {
    float p = w.innerProduct(x);
    return (float)Math.exp(p) / (1.0f + (float)Math.exp(p));
  }

  void svmPredictionsOnTestSet(List<LabeledVector> test_data,
      MutableVector w,
      List<Float> predictions) {
    predictions.clear();
    int size = test_data.size();
    for (int i = 0; i < size; ++i) {
      predictions.add(w.innerProduct(test_data.get(i)));
    }
  }

  void logisticPredictionsOnTestSet(List<LabeledVector> test_data,
      MutableVector w,
      List<Float> predictions) {
    predictions.clear();
    int size = test_data.size();
    for (int i = 0; i < size; ++i) {
      predictions.add(singleLogisticPrediction(test_data.get(i),
          w));
    }
  }

  float svmObjective(List<LabeledVector> data_set,
      MutableVector w,
      float lambda) {
    List<Float> predictions = new ArrayList<Float>();
    svmPredictionsOnTestSet(data_set, w, predictions);
    float objective = (float)w.getSquaredNorm() * lambda / 2.0f;
    for (int i = 0; i < data_set.size(); ++i) {
      float loss_i = 1.0f - (predictions.get(i) * data_set.get(i).getLabel());
      float incremental_loss = (loss_i < 0.0f) ? 
          0.0f : loss_i / data_set.size();
      objective += incremental_loss;
    }
    return objective;
  }
}
