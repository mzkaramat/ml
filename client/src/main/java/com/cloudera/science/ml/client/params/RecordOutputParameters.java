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

import java.util.Locale;

import org.apache.crunch.PCollection;
import org.apache.crunch.Target.WriteMode;
import org.apache.crunch.fn.IdentityFn;
import org.apache.crunch.io.At;
import org.apache.crunch.io.To;
import org.apache.crunch.types.avro.AvroType;
import org.apache.crunch.types.avro.AvroTypeFamily;
import org.apache.crunch.types.writable.WritableTypeFamily;

import com.beust.jcommander.Parameter;
import com.cloudera.science.ml.client.cmd.CommandException;
import com.cloudera.science.ml.core.records.Record;
import com.cloudera.science.ml.core.records.Spec;
import com.cloudera.science.ml.parallel.types.MLRecords;

/**
 * Specifies the formats that can be used with record outputs.
 */
public class RecordOutputParameters {

  public static final String FORMAT_AVRO = "avro";
  public static final String FORMAT_SEQ = "seq";
  public static final String FORMAT_CSV = "csv";
  
  @Parameter(names = "--output-type", required=true,
      description = "The format for the output records, either 'avro', 'seq', or 'csv'")
  private String outputType;

  @Parameter(names = "--csv-delim",
      description = "For writing records as CSV files, the delimiter to use")
  private String delim = ",";
  
  public void writeRecords(PCollection<Record> records, String outputPath) {
    writeRecords(records, null, outputPath);
  }
  
  public void writeRecords(PCollection<Record> records, Spec spec, String output) {
    outputType = outputType.toLowerCase(Locale.ENGLISH);
    if (FORMAT_CSV.equals(outputType)) {
      records = records.parallelDo(IdentityFn.<Record>getInstance(), MLRecords.csvRecord(
          WritableTypeFamily.getInstance(), delim));
      records.write(To.textFile(output), WriteMode.OVERWRITE);
    } else if (FORMAT_AVRO.equals(outputType)) {
      if (AvroTypeFamily.getInstance() != records.getTypeFamily()) {
        if (spec == null) {
          throw new CommandException("Spec required for Avro output conversion");
        }
        records = records.parallelDo(IdentityFn.<Record>getInstance(), MLRecords.record(spec));
      } else {
        records.write(At.avroFile(output, (AvroType<Record>) records.getPType()), WriteMode.OVERWRITE);
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
}
