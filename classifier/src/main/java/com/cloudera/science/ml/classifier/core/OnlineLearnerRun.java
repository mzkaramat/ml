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

import java.io.Serializable;

/**
 * A classification model along with the parameters and data that were used to
 * train it.
 */
public class OnlineLearnerRun implements Serializable {
  private Classifier classifier;
  private OnlineLearnerParams params;
  private int fold;
  private int partition;
  private int paramsVersion;
  
  public OnlineLearnerRun(Classifier classifier, OnlineLearnerParams params,
      int fold, int partition, int paramsVersion) {
    this.classifier = classifier;
    this.params = params;
    this.paramsVersion = paramsVersion;
    this.fold = fold;
    this.partition = partition;
  }
  
  public int getFold() {
    return fold;
  }
  
  public int getPartition() {
    return partition;
  }
  
  /**
   * When different parameters are tried in different runs, an identifier for
   * the set of parameters used for this run.
   */
  public int getParamsVersion() {
    return paramsVersion;
  }
  
  public Classifier getClassifier() {
    return classifier;
  }
  
  public OnlineLearnerParams getParams() {
    return params;
  }
  
  @Override
  public String toString() {
    return "[classifier=" + classifier.getClass().getSimpleName() + ", params="
        + params.toString() + ", fold=" + fold + ", partition=" + partition + "]";
  }
}
