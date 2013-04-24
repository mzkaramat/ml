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
package com.cloudera.science.ml.parallel.fn;

import java.util.Iterator;
import java.util.Map;
import java.util.SortedMap;

import org.apache.crunch.MapFn;
import org.apache.mahout.math.NamedVector;
import org.apache.mahout.math.Vector;

import com.google.common.collect.Maps;

/**
 * Converts an input {@code Vector} type into an SVMLight-formatted string. For
 * {@code NamedVector} instances, the name of the vector will be written at the
 * head of the input.
 */
public class SvmLightFn<V extends Vector> extends MapFn<V, String> {
  @Override
  public String map(V v) {
    StringBuilder sb = new StringBuilder();
    if (v instanceof NamedVector) {
      sb.append(((NamedVector) v).getName()).append(' ');
    }
    Iterator<Vector.Element> iter = v.iterateNonZero();
    if (v.isSequentialAccess()) {
      if (iter.hasNext()) {
        Vector.Element first = iter.next();
        sb.append(first.index()).append(':').append(first.get());
      }
      while (iter.hasNext()) {
        Vector.Element e = iter.next();
        sb.append(' ').append(e.index()).append(':').append(e.get());
      }
    } else {
      SortedMap<Integer, Double> values = Maps.newTreeMap();
      while (iter.hasNext()) {
        Vector.Element e = iter.next();
        values.put(e.index(), e.get());
      }
      boolean first = true;
      for (Map.Entry<Integer, Double> e : values.entrySet()) {
        if (first) {
          first = false;
        } else {
          sb.append(' ');
        }
        sb.append(e.getKey()).append(':').append(e.getValue());
      }
    }
    return sb.toString();
  }
}
