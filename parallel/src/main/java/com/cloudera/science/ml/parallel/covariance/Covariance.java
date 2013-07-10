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
package com.cloudera.science.ml.parallel.covariance;

import static org.apache.crunch.types.avro.Avros.*;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Maps;
import org.apache.crunch.Aggregator;
import org.apache.crunch.DoFn;
import org.apache.crunch.Emitter;
import org.apache.crunch.PCollection;
import org.apache.crunch.PTable;
import org.apache.crunch.Pair;
import org.apache.crunch.types.PType;
import org.apache.hadoop.conf.Configuration;
import org.apache.mahout.math.Vector;

import java.util.Map;
import java.util.Set;

public class Covariance {

  private static final PType<Index> INDEX_PTYPE = reflects(Index.class);
  private static final PType<CoMoment> COMOMENT_PTYPE = reflects(CoMoment.class);

  public static PTable<Index, CoMoment> cov(PCollection<Vector> matrix) {
    return matrix.parallelDo(new CoMomentMapDoFn(), tableOf(INDEX_PTYPE, COMOMENT_PTYPE))
        .groupByKey()
        .combineValues(new CoMomentAggregator());
  }

  public static <K> PTable<Pair<K, Index>, CoMoment> cov(PTable<K, Vector> vectors) {
    PType<Pair<K, Index>> keyType = pairs(vectors.getKeyType(), INDEX_PTYPE);
    return vectors.parallelDo(new CoMomentKeyFn<K>(vectors.getKeyType()), tableOf(keyType, COMOMENT_PTYPE))
        .groupByKey()
        .combineValues(new CoMomentAggregator());
  }

  private static class CoMomentMapDoFn extends DoFn<Vector, Pair<Index, CoMoment>> {
    private transient CoMomentTracker tracker;

    @Override
    public void initialize() {
      if (tracker == null) {
        tracker = new CoMomentTracker();
      }
      tracker.reset();
    }

    @Override
    public void process(Vector vector, Emitter<Pair<Index, CoMoment>> emitter) {
      tracker.update(vector);
    }

    @Override
    public void cleanup(Emitter<Pair<Index, CoMoment>> emitter) {
      for (Map.Entry<Index, CoMoment> e : tracker.entrySet()) {
        emitter.emit(Pair.of(e.getKey(), e.getValue()));
      }
    }
  }

  private static class CoMomentKeyFn<K> extends DoFn<Pair<K, Vector>, Pair<Pair<K, Index>, CoMoment>> {
    private Map<K, CoMomentTracker> trackers = Maps.newHashMap();
    private PType<K> keyType;

    public CoMomentKeyFn(PType<K> keyType) {
      this.keyType = keyType;
    }

    @Override
    public void initialize() {
      trackers.clear();
      keyType.initialize(getConfiguration());
    }

    @Override
    public void process(Pair<K, Vector> input, Emitter<Pair<Pair<K, Index>, CoMoment>> emitter) {
      CoMomentTracker tracker = trackers.get(input.first());
      if (tracker == null) {
        tracker = new CoMomentTracker();
        trackers.put(keyType.getDetachedValue(input.first()), tracker);
      }
      tracker.update(input.second());
    }

    @Override
    public void cleanup(Emitter<Pair<Pair<K, Index>, CoMoment>> emitter) {
      for (Map.Entry<K, CoMomentTracker> e : trackers.entrySet()) {
        for (Map.Entry<Index, CoMoment> f : e.getValue().entrySet()) {
          emitter.emit(Pair.of(Pair.of(e.getKey(), f.getKey()), f.getValue()));
        }
      }
    }
  }

  private static class CoMomentTracker {
    private Map<Index, CoMoment> cache = Maps.newHashMap();

    public void reset() {
      cache.clear();
    }

    public void update(Vector v) {
      for (int i = 0; i < v.size(); i++) {
        double vi = v.getQuick(i);
        for (int j = i; j < v.size(); j++) {
          Index idx = new Index(i, j);
          CoMoment cm = cache.get(idx);
          if (cm == null) {
            cm = new CoMoment();
            cache.put(idx, cm);
          }
          cm.update(vi, v.getQuick(j));
        }
      }
    }

    public Set<Map.Entry<Index, CoMoment>> entrySet() {
      return cache.entrySet();
    }
  }

  private static class CoMomentAggregator implements Aggregator<CoMoment> {
    private CoMoment cm;
    @Override
    public void initialize(Configuration entries) {
      reset();
    }

    @Override
    public void reset() {
      cm = new CoMoment();
    }

    @Override
    public void update(CoMoment coMoment) {
      cm = cm.merge(coMoment);
    }

    @Override
    public Iterable<CoMoment> results() {
      return ImmutableList.of(cm);
    }
  }

  private Covariance() {
  }
}
