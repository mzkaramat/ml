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
package com.cloudera.science.ml.kmeans.parallel;

import org.apache.crunch.MapFn;

import com.cloudera.science.ml.core.vectors.Centers;
import com.cloudera.science.ml.core.vectors.LabeledVector;
import com.cloudera.science.ml.core.vectors.Vectors;

class TriangleClusterFeaturesFn extends MapFn<LabeledVector, LabeledVector> {
  private Centers centers;
  private LabeledVector outVec;
  
  public TriangleClusterFeaturesFn(Centers centers) {
    this.centers = centers;
  }
  
  @Override
  public LabeledVector map(LabeledVector vec) {
    if (outVec == null) {
      outVec = new LabeledVector(Vectors.dense(centers.size()), Double.NaN);
    }
    outVec.setLabel(vec.getLabel());
    double sumDist = 0.0;
    for (int i = 0; i < centers.size(); i++) {
      double dist = vec.getDistanceSquared(centers.get(i));
      sumDist += dist;
      outVec.set(i, dist);
    }
    
    double avgDist = sumDist / centers.size();
    for (int i = 0; i < centers.size(); i++) {
      outVec.set(i, Math.max(0, avgDist - outVec.get(i)));
    }
    return outVec;
  }
}