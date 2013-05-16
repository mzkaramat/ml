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

import java.io.Serializable;

public class ClassifierParams implements Serializable {
  private LearnerType learnerType;
  private LoopType loopType;
  private EtaType etaType;
  private float lambda;
  private float c;
  private int numIters;
  private int numFeatures;

  public ClassifierParams(LearnerType learnerType, LoopType loopType,
      EtaType etaType, float lambda, float c, int numIters, int numFeatures) {
    this.learnerType = learnerType;
    this.loopType = loopType;
    this.etaType = etaType ;
    this.lambda = lambda;
    this.c = c;
    this.numIters = numIters;
    this.numFeatures = numFeatures;
  }
  
  public LearnerType getLearnerType() {
    return learnerType;
  }
  
  public LoopType getLoopType() {
    return loopType;
  }
  
  public EtaType getEtaType() {
    return etaType;
  }
  
  public float getLambda() {
    return lambda;
  }
  
  public float getC() {
    return c;
  }
  
  public int getNumIters() {
    return numIters;
  }
  
  public int getNumFeatures() {
    return numFeatures;
  }
}
