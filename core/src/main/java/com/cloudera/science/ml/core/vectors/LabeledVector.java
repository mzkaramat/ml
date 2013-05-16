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

package com.cloudera.science.ml.core.vectors;

import org.apache.mahout.math.Vector;

public class LabeledVector {
  private double squaredNorm;
  private Vector vector;
  private double label;
  
  public LabeledVector(Vector vector, double label) {
    this.vector = vector;
    this.label = label;
  }

  public int getDimension() {
    return vector.size();
  }
  
  public Vector getVector() {
    return vector;
  }
  
  public float getSquaredNorm() {
    //TODO: lazily calculate this
    return (float)squaredNorm;
  }
    
  public float getLabel() {
    return (float)label;
  }
}
