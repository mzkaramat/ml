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

import java.io.Serializable;

public class ClusterKey implements Serializable {

  private int clusterId;
  private int centerId;

  public ClusterKey() { }

  public ClusterKey(int clusterId, int centerId) {
    this.clusterId = clusterId;
    this.centerId = centerId;
  }

  public int getClusterId() {
    return clusterId;
  }

  public int getCenterId() {
    return centerId;
  }

  @Override
  public int hashCode() {
    return 17 * clusterId + 37 * centerId;
  }

  @Override
  public boolean equals(Object other) {
    if (!(other instanceof ClusterKey)) {
      return false;
    }
    ClusterKey ck = (ClusterKey) other;
    return clusterId == ck.clusterId && centerId == ck.centerId;
  }
}
