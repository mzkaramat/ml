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
package com.cloudera.science.ml.kmeans.core;

import java.util.Collection;
import java.util.List;
import java.util.Map;

import org.apache.mahout.math.Vector;

import com.cloudera.science.ml.core.vectors.Centers;
import com.cloudera.science.ml.core.vectors.Weighted;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;

/**
 *
 */
public class LloydsUpdateStrategy implements KMeansUpdateStrategy {

  private final int numIterations;
  
  public LloydsUpdateStrategy(int numIterations) {
    this.numIterations = numIterations;
  }
  
  @Override
  public <V extends Vector> Centers update(List<Weighted<V>> points, Centers centers) {
    for (int iter = 0; iter < numIterations; iter++) {
      Map<Integer, List<Weighted<V>>> assignments = Maps.newHashMap();
      for (int i = 0; i < centers.size(); i++) {
        assignments.put(i, Lists.<Weighted<V>>newArrayList());
      }
      for (Weighted<V> weightedVec : points) {
        assignments.get(centers.indexOfClosest(weightedVec.thing())).add(weightedVec);
      }
      List<Vector> centroids = Lists.newArrayList();
      for (Map.Entry<Integer, List<Weighted<V>>> e : assignments.entrySet()) {
        if (e.getValue().isEmpty()) {
          centroids.add(centers.get(e.getKey())); // fix the no-op center
        } else {
          centroids.add(centroid(e.getValue()));
        }
      }
      centers = new Centers(centroids);
    }
    return centers;
  }

  /**
   * Compute the {@code Vector} that is the centroid of the given weighted points.
   * 
   * @param points The weighted points
   * @return The centroid of the weighted points
   */
  public <V extends Vector> Vector centroid(Collection<Weighted<V>> points) {
    Vector center = null;
    double sz = 0.0;
    for (Weighted<V> v : points) {
      Vector weighted = v.thing().times(v.weight());
      if (center == null) {
        center = weighted;
      } else {
        center = center.plus(weighted);
      }
      sz += v.weight();
    }
    return center.divide(sz);
  }  
}
