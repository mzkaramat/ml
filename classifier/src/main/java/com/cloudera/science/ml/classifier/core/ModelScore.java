/**
 * Copyright (c) 2012, Cloudera, Inc. All Rights Reserved.
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

import java.text.DecimalFormat;

public class ModelScore {
  private int trueNegatives;
  private int falseNegatives;
  private int truePositives;
  private int falsePositives;
  
  public ModelScore(int trueNegatives, int falseNegatives, int truePositives,
      int falsePositives) {
    this.trueNegatives = trueNegatives;
    this.falseNegatives = falseNegatives;
    this.truePositives = truePositives;
    this.falsePositives = falsePositives;
  }
  
  public int getTrueNegatives() {
    return trueNegatives;
  }
  
  public int getFalseNegatives() {
    return falseNegatives;
  }
  
  public int getTruePositives() {
    return truePositives;
  }
  
  public int getFalsePositives() {
    return falsePositives;
  }
  
  public double getPrecision() {
    return (double) truePositives / (truePositives + falsePositives);
  }
  
  public double getRecall() {
    return (double) truePositives / (truePositives + falseNegatives);
  }
  
  public double getAccuracy() {
    return (double) (truePositives + trueNegatives) /
        (truePositives + trueNegatives + falseNegatives + falsePositives);
  }

  public void merge(ModelScore other) {
    falseNegatives += other.falseNegatives;
    trueNegatives += other.trueNegatives;
    falsePositives += other.falsePositives;
    truePositives += other.truePositives;
  }
  
  public String toString() {
    DecimalFormat df = new DecimalFormat("#.##");
    return "[TN= " + trueNegatives + ", FN=" + falseNegatives + ", TP="
        + truePositives + ", FP=" + falsePositives + ", accuracy=" +
        df.format(getAccuracy()) + ", precision=" + df.format(getPrecision()) +
        ", recall=" + df.format(getRecall()) + "]";
  }
}
