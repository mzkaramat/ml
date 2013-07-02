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
package com.cloudera.science.ml.parallel.crossfold;

import java.util.Collection;
import java.util.HashSet;

import junit.framework.Assert;

import org.apache.crunch.PCollection;
import org.apache.crunch.Pair;
import org.apache.crunch.impl.mem.MemPipeline;
import org.apache.crunch.materialize.pobject.CollectionPObject;
import org.apache.crunch.types.PTypeFamily;
import org.apache.crunch.types.avro.Avros;
import org.junit.Test;

import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.Multimap;

public class TestCrossfoldFn {
  private static final long seed = 1729L;
  
  @Test
  public void testCrossfoldFn() {
    final int numFolds = 3;
    CrossfoldFn<String> crossfoldFn = new CrossfoldFn<String>(numFolds, seed);
    String[] strings = {"jim", "corbett", "william", "shirer", "raymond", "chandler"};
    PCollection<String> pstrings = MemPipeline.typedCollectionOf(Avros.strings(),
        strings);
    PTypeFamily ptf = pstrings.getTypeFamily();
    PCollection<Pair<Integer, String>> pcrossfolded =
        pstrings.parallelDo(crossfoldFn, ptf.pairs(ptf.ints(), ptf.strings()));
    
    Collection<Pair<Integer, String>> crossfolded =
        new CollectionPObject<Pair<Integer, String>>(pcrossfolded).getValue();
    
    Assert.assertEquals((numFolds-1) * strings.length, crossfolded.size());
    Multimap<String, Integer> map = pairCollectionToInvertedMultimap(crossfolded);
    Assert.assertEquals(strings.length, map.keySet().size());
    for (String item : map.keys()) {
      Collection<Integer> folds = map.get(item);
      Assert.assertEquals(numFolds-1, folds.size());
      // Assert all different
      HashSet<Integer> uniques = new HashSet<Integer>();
      uniques.addAll(folds);
      Assert.assertEquals(folds.size(), uniques.size());
    }
  }
  
  @Test
  public void testCrossfoldFnWithCrossfold() {
    final int numFolds = 3;
    CrossfoldFn<String> crossfoldFn = new CrossfoldFn<String>(numFolds, seed);
    String[] strings = {"jim", "corbett", "william", "shirer", "raymond", "chandler",
        "jk", "rowling", "salman", "rushdie", "thomas", "pynchon", "david",
        "foster", "wallace"};
    PCollection<String> pstrings = MemPipeline.typedCollectionOf(Avros.strings(),
        strings);
    PTypeFamily ptf = pstrings.getTypeFamily();
    PCollection<Pair<Integer, String>> pcrossfolded =
        pstrings.parallelDo(crossfoldFn, ptf.pairs(ptf.ints(), ptf.strings()));
    
    Crossfold crossfold = new Crossfold(numFolds, seed);
    PCollection<Pair<Integer, String>> pcrossfolds = crossfold.apply(pstrings);
    
    Collection<Pair<Integer, String>> crossfolded =
        new CollectionPObject<Pair<Integer, String>>(pcrossfolded).getValue();
    Collection<Pair<Integer, String>> crossfolds =
        new CollectionPObject<Pair<Integer, String>>(pcrossfolds).getValue();
 
    Assert.assertEquals((numFolds-1) * strings.length, crossfolded.size());
    Assert.assertEquals(strings.length, crossfolds.size());
    
    Multimap<Integer, String> crossfoldedMap = pairCollectionToMultimap(crossfolded);
    Multimap<Integer, String> crossfoldsMap = pairCollectionToMultimap(crossfolds);
    Assert.assertEquals(crossfoldedMap.keySet().size(), crossfoldsMap.keySet().size());
    Assert.assertEquals(numFolds, crossfoldsMap.keySet().size());
    
    for (Integer fold : crossfoldedMap.keys()) {
      Collection<String> crossfoldedItems = crossfoldedMap.get(fold);
      Collection<String> crossfoldsItems = crossfoldsMap.get(fold);
      Assert.assertEquals(strings.length, crossfoldedItems.size() + crossfoldsItems.size());
      HashSet<String> uniques = new HashSet<String>();
      uniques.addAll(crossfoldsItems);
      uniques.addAll(crossfoldedItems);
      Assert.assertEquals(strings.length, uniques.size());
    }
  }
  
  private <K, V> Multimap<K, V> pairCollectionToMultimap(
      Collection<Pair<K, V>> collection) {
    Multimap<K, V> map = ArrayListMultimap.create();
    for (Pair<K, V> pair : collection) {
      map.put(pair.first(), pair.second());
    }
    return map;
  }
  
  private <K, V> Multimap<K, V> pairCollectionToInvertedMultimap(
      Collection<Pair<V, K>> collection) {
    Multimap<K, V> map = ArrayListMultimap.create();
    for (Pair<V, K> pair : collection) {
      map.put(pair.second(), pair.first());
    }
    return map;
  }
}
