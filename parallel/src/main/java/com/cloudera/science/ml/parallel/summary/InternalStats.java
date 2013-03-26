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

import java.util.Map;
import java.util.Set;

import org.apache.crunch.Aggregator;
import org.apache.crunch.fn.Aggregators.SimpleAggregator;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;

/**
 *
 */
class InternalStats {

  public static final Aggregator<InternalStats> AGGREGATOR = new SimpleAggregator<InternalStats>() {
    private InternalStats agg;
    @Override
    public void reset() {
      agg = new InternalStats();
    }

    @Override
    public Iterable<InternalStats> results() {
      return ImmutableList.of(agg);
    }

    @Override
    public void update(InternalStats other) {
      agg.merge(other);
    }
  };
  
  private InternalNumeric internalNumeric;
  private Map<String, Entry> histogram;

  public InternalStats() {
  }
  
  public SummaryStats toSummaryStats(String name, long recordCount) {
    if (internalNumeric == null) {
      if (histogram == null) {
        return new SummaryStats(name);
      } else {
        return new SummaryStats(name, histogram);
      }
    } else {
      return new SummaryStats(name, internalNumeric.toNumeric(recordCount));
    }
  }
  
  private InternalNumeric internalNumeric() {
    if (internalNumeric == null) {
      internalNumeric = new InternalNumeric();
    }
    return internalNumeric;
  }
  
  private Map<String, Entry> histogram() {
    if (histogram == null) {
      histogram = Maps.newHashMap();
    }
    return histogram;
  }
  
  public void addSymbol(String symbol) {
    Map<String, Entry> h = histogram();
    Entry entry = h.get(symbol);
    if (entry == null) {
      entry = new Entry(h.size()).inc(); // init with count = 1
      h.put(symbol, entry);
    } else {
      entry.inc();
    }
  }
  
  public void addNumeric(double value) {
    internalNumeric().update(value);
  }
  
  public void merge(InternalStats other) {
    if (other.internalNumeric != null) {
      internalNumeric().merge(other.internalNumeric);
    } else {
      Map<String, Entry> entries = histogram();
      Map<String, Entry> merged = Maps.newTreeMap();
      Set<String> keys = Sets.newTreeSet(
          Sets.union(entries.keySet(), other.histogram().keySet()));
      for (String key : keys) {
        Entry e = entries.get(key);
        Entry entry = other.histogram().get(key);
        Entry newEntry = new Entry(merged.size());
        if (e != null) {
          newEntry.inc(e.count);
        }
        if (entry != null) {
          newEntry.inc(entry.count);
        }
        merged.put(key, newEntry);
      }
      entries.clear();
      entries.putAll(merged);
    }
  }
}
