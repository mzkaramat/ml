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
package com.cloudera.science.ml.hcatalog;

import org.apache.crunch.MapFn;
import org.apache.crunch.types.PType;
import org.apache.crunch.types.writable.Writables;
import org.apache.hadoop.hive.conf.HiveConf;
import org.apache.hadoop.hive.metastore.HiveMetaStoreClient;
import org.apache.hadoop.hive.ql.metadata.Table;
import org.apache.hcatalog.common.HCatException;
import org.apache.hcatalog.common.HCatUtil;
import org.apache.hcatalog.data.HCatRecord;
import org.apache.hcatalog.data.schema.HCatSchema;

import com.cloudera.science.ml.core.records.Record;
import com.cloudera.science.ml.core.records.RecordSpec;

public final class HCatalog {

  private static HiveMetaStoreClient CLIENT_INSTANCE = null;
  
  private static synchronized HiveMetaStoreClient getClientInstance() {
    if (CLIENT_INSTANCE == null) {
      try {
        CLIENT_INSTANCE = HCatUtil.getHiveClient(new HiveConf());
      } catch (Exception e) {
        throw new RuntimeException("Could not connect to Hive", e);
      }
    }
    return CLIENT_INSTANCE;
  }
  
  public static Table getTable(String dbName, String tableName) {
    HiveMetaStoreClient client = getClientInstance();
    Table table;
    try {
      table = HCatUtil.getTable(client, dbName, tableName);
    } catch (Exception e) {
      throw new RuntimeException("Hive table lookup exception", e);
    }
    
    if (table == null) {
      throw new IllegalStateException("Could not find info for table: " + tableName);
    }
    return table;
  }
  
  public static RecordSpec getSpec(String dbName, String tableName) {
    Table table = getTable(dbName, tableName);
    try {
      return new HCatalogSpec(HCatUtil.extractSchema(table));
    } catch (HCatException e) {
      throw new RuntimeException("HCatalog schema extraction error", e);
    }
  }
  
  public static PType<Record> records(HCatSchema dataSchema) {
    return Writables.derived(Record.class, new HCatInFn(dataSchema),
        new HCatOutFn(), Writables.writables(HCatRecord.class));
  }

  private static class HCatInFn extends MapFn<HCatRecord, Record> {
    private final HCatSchema dataSchema;
    
    HCatInFn(HCatSchema dataSchema) {
      this.dataSchema = dataSchema;
    }
    
    @Override
    public Record map(HCatRecord impl) {
      return new HCatalogRecord(impl, dataSchema);
    }
  }
  
  private static class HCatOutFn extends MapFn<Record, HCatRecord> {
    @Override
    public HCatRecord map(Record rec) {
      if (rec instanceof HCatalogRecord) {
        return ((HCatalogRecord) rec).getImpl();
      } else {
        throw new UnsupportedOperationException("HCatOut does not support generic records yet");
      }
    }
  }
  
  private HCatalog() { }
}
