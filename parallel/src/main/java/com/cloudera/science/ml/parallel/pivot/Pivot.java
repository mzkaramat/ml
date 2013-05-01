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
package com.cloudera.science.ml.parallel.pivot;

import java.util.List;
import java.util.Map;

import org.apache.crunch.DoFn;
import org.apache.crunch.Emitter;
import org.apache.crunch.MapFn;
import org.apache.crunch.Pair;
import org.apache.crunch.types.PTableType;
import org.apache.crunch.types.avro.Avros;

import com.cloudera.science.ml.core.records.FieldSpec;
import com.cloudera.science.ml.core.records.Record;
import com.cloudera.science.ml.core.records.RecordSpec;
import com.cloudera.science.ml.core.records.SimpleRecord;
import com.cloudera.science.ml.core.records.Spec;
import com.cloudera.science.ml.core.summary.Summary;
import com.cloudera.science.ml.core.summary.SummaryStats;
import com.cloudera.science.ml.parallel.records.Records;
import com.cloudera.science.ml.parallel.records.SummarizedRecords;
import com.cloudera.science.ml.parallel.types.MLRecords;
import com.google.common.collect.Maps;

public final class Pivot {

  private Pivot() {
  }

  public enum Agg { SUM, MEAN }

  private static Spec createSpec(Spec spec, List<Integer> groupColumns) {
    RecordSpec.Builder b = RecordSpec.builder();
    for (Integer c : groupColumns) {
      FieldSpec f = spec.getField(c);
      b.add(f.name(), f.spec());
    }
    return b.build();
  }
  
  public static Records pivot(SummarizedRecords records,
                              List<Integer> groupColumns,
                              int attributeColumn,
                              List<Integer> valueColumns,
                              Agg agg) {
    Summary summary = records.getSummary();
    Spec keySpec = createSpec(records.getSpec(), groupColumns);
    PTableType<Record, Map<String, Stat>> ptt = Avros.tableOf(
        MLRecords.record(keySpec),
        Avros.maps(Avros.reflects(Stat.class)));

    RecordSpec.Builder b = RecordSpec.builder(keySpec);
    SummaryStats attrStats = summary.getStats(attributeColumn);
    if (attrStats.isNumeric() || attrStats.numLevels() == 1) {
      throw new IllegalArgumentException("Non-categorical attribute column in pivot op");
    }
    
    List<String> levels = attrStats.getLevels();
    for (Integer valueColumn : valueColumns) {
      SummaryStats valueStats = summary.getStats(valueColumn);
      if (!valueStats.isNumeric()) {
        throw new IllegalArgumentException("Non-numeric value column in pivot op");
      }
      String valueName = summary.getSpec().getField(valueColumn).name();
      for (String level : levels) {
        b.addDouble(String.format("%s_%s", valueName, level));
      }
    }

    Spec outSpec = b.build();
    return new Records(records.get().parallelDo("pivotmap",
        new PivotMapperFn(keySpec, groupColumns, attributeColumn, valueColumns),
        ptt)
        .groupByKey()
        .combineValues(new MapAggregator())
        .parallelDo("makerecord",
            new PivotFinishFn(outSpec, levels, valueColumns.size(), agg),
            MLRecords.record(outSpec)), outSpec);
  }
  
  private static class PivotMapperFn extends DoFn<Record, Pair<Record, Map<String, Stat>>> {

    private final Spec spec;
    private final List<Integer> groupColumns;
    private final int attributeColumn;
    private final List<Integer> valueColumns;
    private final Map<Record, Map<String, Stat>> cache;
    private int cacheAdds = 0;
    
    private PivotMapperFn(Spec spec, List<Integer> groupColumns, int attributeColumn,
        List<Integer> valueColumns) {
      this.spec = spec;
      this.groupColumns = groupColumns;
      this.attributeColumn = attributeColumn;
      this.valueColumns = valueColumns;
      this.cache = Maps.newHashMap();
    }
    
    @Override
    public void process(Record r, Emitter<Pair<Record, Map<String, Stat>>> emitter) {
      Record key = new SimpleRecord(spec);
      for (int i = 0; i < groupColumns.size(); i++) {
        key.set(i, r.get(groupColumns.get(i)));
      }

      Map<String, Stat> ss = cache.get(key);
      if (ss == null) {
        ss = Maps.newHashMap();
        cache.put(key, ss);
      }
      
      String level = r.getAsString(attributeColumn);
      Stat stat = ss.get(level);
      if (stat == null) {
        stat = new Stat(valueColumns.size());
        ss.put(level, stat);
        cacheAdds++;
        System.out.println("Created new stat");
      } else {
        System.out.println("Found existing stat");
      }
      
      for (int i = 0; i < valueColumns.size(); i++) {
        stat.inc(i, r.getAsDouble(valueColumns.get(i)));
      }
      
      if (cacheAdds > 10000) { //TODO parameterize
        cleanup(emitter);
      }
    }
    
    @Override
    public void cleanup(Emitter<Pair<Record, Map<String, Stat>>> emitter) {
      System.out.println("Cleaning up...");
      for (Map.Entry<Record, Map<String, Stat>> e : cache.entrySet()) {
        emitter.emit(Pair.of(e.getKey(), e.getValue()));
      }
      cache.clear();
      cacheAdds = 0;
    }
  }
  
  private static class PivotFinishFn extends MapFn<Pair<Record, Map<String, Stat>>, Record> {
    private final Spec spec;
    private final List<String> levels;
    private final int numValues;
    private final Agg agg;
    
    private PivotFinishFn(Spec spec, List<String> levels, int numValues, Agg agg) {
      this.spec = spec;
      this.levels = levels;
      this.numValues = numValues;
      this.agg = agg;
    }
    
    @Override
    public Record map(Pair<Record, Map<String, Stat>> p) {
      Record r = new SimpleRecord(spec);
      int index = p.first().getSpec().size();
      for (int i = 0; i < index; i++) {
        r.set(i, p.first().get(i));
      }
      for (int i = 0; i < numValues; i++) {
        for (int j = 0; j < levels.size(); j++) {
          Stat ss = p.second().get(levels.get(j));
          double stat = 0.0;
          if (ss != null) {
            if (agg == Agg.MEAN) {
              stat = ss.getSum(i) / ss.getCount(i);
            } else {
              stat = ss.getSum(i);
            }
          }
          r.set(index, stat);
          index++;
        }
      }
      return r;
    }
  }

}
