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
package com.cloudera.science.ml.client.params;

import java.io.IOException;
import java.util.List;
import java.util.Locale;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.crunch.PCollection;
import org.apache.crunch.Target.WriteMode;
import org.apache.crunch.fn.IdentityFn;
import org.apache.crunch.io.At;
import org.apache.crunch.io.To;
import org.apache.crunch.types.avro.AvroType;
import org.apache.crunch.types.avro.AvroTypeFamily;
import org.apache.crunch.types.writable.WritableTypeFamily;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.hive.metastore.TableType;
import org.apache.hadoop.hive.metastore.api.FieldSchema;
import org.apache.hadoop.hive.ql.metadata.Table;

import com.beust.jcommander.Parameter;
import com.cloudera.science.ml.client.cmd.CommandException;
import com.cloudera.science.ml.core.records.FieldSpec;
import com.cloudera.science.ml.core.records.Record;
import com.cloudera.science.ml.core.records.Spec;
import com.cloudera.science.ml.core.records.avro.Spec2Schema;
import com.cloudera.science.ml.hcatalog.HCatalog;
import com.cloudera.science.ml.parallel.types.MLRecords;
import com.google.common.collect.Lists;

/**
 * Specifies the formats that can be used with record outputs.
 */
public class RecordOutputParameters {

  private static final Log LOG = LogFactory.getLog(RecordOutputParameters.class);
  
  public static final String FORMAT_AVRO = "avro";
  public static final String FORMAT_SEQ = "seq";
  public static final String FORMAT_CSV = "csv";
  
  @Parameter(names = "--output-type", required=true,
      description = "The format for the output records, either 'avro', 'seq', or 'csv'")
  private String outputType;

  @Parameter(names = "--csv-delim",
      description = "For writing records as CSV files, the delimiter to use")
  private String delim = ",";
  
  @Parameter(names = "--output-table",
      description = "Creates a new external Hive table with the given name for the output")
  private String hiveStr;
  
  public void writeRecords(PCollection<Record> records, String outputPath) throws IOException {
    writeRecords(records, null, outputPath);
  }
  
  public void writeRecords(PCollection<Record> records, Spec spec, String output) throws IOException {
    outputType = outputType.toLowerCase(Locale.ENGLISH);
    if (FORMAT_CSV.equals(outputType)) {
      records = records.parallelDo(IdentityFn.<Record>getInstance(), MLRecords.csvRecord(
          WritableTypeFamily.getInstance(), delim));
      records.write(To.textFile(output), WriteMode.OVERWRITE);
      if (spec != null) {
        createHiveTable(spec, output);
      }
    } else if (FORMAT_AVRO.equals(outputType)) {
      if (AvroTypeFamily.getInstance() != records.getTypeFamily()) {
        if (spec == null) {
          throw new CommandException("Spec required for Avro output conversion");
        }
        records = records.parallelDo(IdentityFn.<Record>getInstance(), MLRecords.record(spec));
      } else {
        records.write(At.avroFile(output, (AvroType<Record>) records.getPType()), WriteMode.OVERWRITE);
      }
      if (spec != null) {
        createHiveTable(spec, output);
      }
    } else if (FORMAT_SEQ.equals(outputType)) {
      if (WritableTypeFamily.getInstance() != records.getTypeFamily()) {
        throw new CommandException("SequenceFile record outputs must have a Writable serialization type");
      }
      records.write(To.sequenceFile(output), WriteMode.OVERWRITE);
    } else {
      throw new CommandException("Unknown output type: " + outputType);
    }
  }
  
  private void createHiveTable(Spec spec, String outputPath) throws IOException {
    if (hiveStr != null) {
      String dbName = HCatalog.getDbName(hiveStr);
      String tblName = HCatalog.getTableName(hiveStr);
      if (HCatalog.tableExists(dbName, tblName)) {
        LOG.warn("Hive table named " + hiveStr + " already exists");
        return;
      }
      LOG.info("Creating an external Hive table named: " + hiveStr);
      Table tbl = new Table(dbName, tblName);
      tbl.setTableType(TableType.EXTERNAL_TABLE);
      Path output = FileSystem.get(new Configuration()).makeQualified(new Path(outputPath));
      tbl.setDataLocation(output.toUri());
      List<FieldSchema> fields = Lists.newArrayList();
      for (int i = 0; i < spec.size(); i++) {
        FieldSpec fs = spec.getField(i);
        FieldSchema hfs = new FieldSchema();
        hfs.setName(fs.name());
        switch (fs.spec().getDataType()) {
        case BOOLEAN:
          hfs.setType("boolean");
          break;
        case INT:
          hfs.setType("int");
          break;
        case LONG:
          hfs.setType("bigint");
          break;
        case DOUBLE:
          hfs.setType("double");
          break;
        case STRING:
          hfs.setType("string");
          break;
        }
        fields.add(hfs);
      }
      tbl.setFields(fields);
      if (FORMAT_AVRO.equals(outputType)) {
        try {
          tbl.setSerializationLib("org.apache.hadoop.hive.serde2.avro.AvroSerDe");
          tbl.setInputFormatClass("org.apache.hadoop.hive.ql.io.avro.AvroContainerInputFormat");
          tbl.setOutputFormatClass("org.apache.hadoop.hive.ql.io.avro.AvroContainerOutputFormat");
          tbl.setProperty("avro.schema.literal", Spec2Schema.create(spec).toString());
        } catch (Exception e) {
          LOG.error("Error configured Hive Avro table, table creation failed", e);
          return;
        }
      } else { // FORMAT_CSV
        try {
          tbl.setProperty("field.delim", delim); // A bit easier, on the whole
          tbl.setInputFormatClass("org.apache.hadoop.mapred.TextInputFormat");
          tbl.setOutputFormatClass("org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat");
        } catch (Exception e) {
          LOG.error("Error configuring Hive for CSV files, table creation failed", e);
          return;
        }
      }
      HCatalog.createTable(tbl);
    }
  }
}
