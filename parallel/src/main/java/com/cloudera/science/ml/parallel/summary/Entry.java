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
package com.cloudera.science.ml.parallel.summary;

import java.io.Serializable;

import com.google.common.primitives.Longs;

class Entry implements Serializable, Comparable<Entry> {

  private int id;
  private long count;
  
  Entry() { }
  
  Entry(int id) {
    this.id = id;
    this.count = 0;
  }
  
  int getID() {
    return id;
  }
  
  long getCount() {
    return count;
  }
  
  public Entry inc() {
    return inc(1L);
  }
  
  public Entry inc(long count) {
    this.count += count;
    return this;
  }

  @Override
  public int compareTo(Entry other) {
    if (id < other.id) {
      return -1;
    }
    if (id > other.id) {
      return 1;
    }
    return 0;
  }
  
  @Override
  public boolean equals(Object o) {
    if (!(o instanceof Entry)) {
      return false;
    }
    Entry other = (Entry) o;
    return id == other.id && count == other.count;
  }
  
  @Override
  public int hashCode() {
    return id ^ Longs.hashCode(count);
  }
  
}