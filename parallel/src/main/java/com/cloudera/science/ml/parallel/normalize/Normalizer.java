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
package com.cloudera.science.ml.parallel.normalize;

import java.io.Serializable;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.crunch.DoFn;
import org.apache.crunch.Emitter;
import org.apache.crunch.PCollection;
import org.apache.crunch.types.PType;
import org.apache.mahout.math.NamedVector;
import org.apache.mahout.math.Vector;

import com.cloudera.science.ml.core.records.Record;
import com.cloudera.science.ml.core.records.vectors.VectorRecord;
import com.cloudera.science.ml.core.summary.Summary;
import com.cloudera.science.ml.core.summary.SummaryStats;
import com.cloudera.science.ml.core.vectors.Vectors;
import com.google.common.collect.Maps;

/**
 * Converts an input {@code Record} into a normalized {@code Vector} in which all categorical columns are
 * converted to indicator variables.
 */
public class Normalizer implements Serializable {

  private static final Log LOG = LogFactory.getLog(Normalizer.class);
  
  private final Summary summary;
  private final int idColumn;
  private final Set<Integer> ignoredColumns;
  private final Transform defaultTransform;
  private final Map<Integer, Transform> transforms;
  private final int expansion;
  private final boolean sparse;
  
  public static Builder builder() { 
    return new Builder();
  }
  
  public static class Builder {
    private Summary s = new Summary();
    private Boolean sparse = null;
    private int idColumn = -1;
    private Transform defaultTransform = Transform.NONE;
    private final Map<Integer, Transform> transforms = Maps.newHashMap();
    
    public Builder summary(Summary s) {
      if (s != null) {
        this.s = s;
        List<SummaryStats> stats = s.getAllStats();
        for (int i = 0; i < stats.size(); i++) {
          SummaryStats ss = stats.get(i);
          if (ss != null && ss.getTransform() != null) {
            transforms.put(i, Transform.forName(ss.getTransform()));
          }
        }
      }
      return this;
    }
    
    public Builder sparse(Boolean sparse) {
      this.sparse = sparse;
      return this;
    }
    
    public Builder idColumn(int idColumn) {
      this.idColumn = idColumn;
      return this;
    }
    
    public Builder defaultTransform(Transform t) {
      this.defaultTransform = t;
      return this;
    }
    
    public Normalizer build() {
      return new Normalizer(s, sparse, idColumn, defaultTransform, transforms);
    }
  }
  
  private Normalizer(Summary summary, Boolean sparse, int idColumn,
      Transform defaultTransform, Map<Integer, Transform> transforms) {
    this.summary = summary;
    this.idColumn = idColumn;
    this.ignoredColumns = summary.getIgnoredColumns();
    this.defaultTransform = defaultTransform;
    this.transforms = transforms;
    this.expansion = -ignoredColumns.size() + summary.getNetLevels() -
        (idColumn >= 0 && !ignoredColumns.contains(idColumn) ? 1 : 0);
    if (sparse == null) {
      this.sparse = expansion > 2 * (summary.getFieldCount() - ignoredColumns.size());
    } else {
      this.sparse = sparse;
    }
  }
  
  public <V extends Vector> PCollection<V> apply(PCollection<Record> records, PType<V> ptype) {
    return records.parallelDo("standardize", new StandardizeFn<V>(), ptype);
  }
  
  private class StandardizeFn<V extends Vector> extends DoFn<Record, V> {
    @Override
    public void process(Record record, Emitter<V> emitter) {
      int len = record.getSpec().size() + expansion;
      Vector v;
      if (record instanceof VectorRecord) {
        v = ((VectorRecord) record).getVector().like();
      } else if (sparse) {
        v = Vectors.sparse(len);
      } else {
        v = Vectors.dense(len);
      }

      int offset = 0;
      for (int i = 0; i < record.getSpec().size(); i++) {
        if (idColumn != i && !ignoredColumns.contains(i)) {
          SummaryStats ss = summary.getStats(i);
          if (ss == null || ss.isEmpty()) {
            v.setQuick(offset, record.getAsDouble(i));
            offset++;
          } else if (ss.isNumeric()) {
            Transform t = defaultTransform;
            if (transforms.containsKey(i)) {
              t = transforms.get(i);
            }
            double raw = record.getAsDouble(i);
            if (Double.isNaN(raw)) {
              LOG.warn(String.format("Missing/non-numeric value encountered for field %d: '%s', skipping...",
                  i, record.getAsString(i)));
              return;
            }
            double n = t.apply(raw, ss) * ss.getScale();
            v.setQuick(offset, n);
            offset++;
          } else {
            int index = ss.index(record.getAsString(i));
            if (index < 0) {
              LOG.warn(String.format("Unknown categorical value encountered for field %d: '%s', skipping...",
                  i, record.getAsString(i)));
              return;
            }
            v.setQuick(offset + index, ss.getScale());
            offset += ss.numLevels();
          }
        }
      }
      
      if (idColumn >= 0) {
        v = new NamedVector(v, record.getAsString(idColumn));
      }
      emitter.emit((V) v);
    }
  }
}
