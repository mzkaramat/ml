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
package com.cloudera.science.ml.parallel.base;

import org.apache.crunch.Pair;

import com.google.common.base.Function;
import com.google.common.collect.Iterables;

/**
 *
 */
public class Pairs {
  public static <K, V> Iterable<K> first(Iterable<Pair<K, V>> iterable) {
    return Iterables.transform(iterable, new Function<Pair<K, V>, K>() {
      @Override
      public K apply(Pair<K, V> p) {
        return p.first();
      }
    });
  }
  
  public static <K, V> Iterable<V> second(Iterable<Pair<K, V>> iterable) {
    return Iterables.transform(iterable, new Function<Pair<K, V>, V>() {
      @Override
      public V apply(Pair<K, V> p) {
        return p.second();
      }
    });
  }
}
