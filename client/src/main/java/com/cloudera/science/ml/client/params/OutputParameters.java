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

import com.beust.jcommander.Parameter;
import com.cloudera.science.ml.client.cmd.CommandException;
import com.cloudera.science.ml.mahout.types.MLWritables;
import com.cloudera.science.ml.parallel.fn.StringifyFn;
import com.cloudera.science.ml.parallel.types.MLAvros;
import org.apache.crunch.PCollection;
import org.apache.crunch.Target;
import org.apache.crunch.Target.WriteMode;
import org.apache.crunch.fn.IdentityFn;
import org.apache.crunch.io.At;
import org.apache.crunch.io.To;
import org.apache.crunch.types.PType;
import org.apache.crunch.types.PTypeFamily;
import org.apache.crunch.types.avro.AvroType;
import org.apache.crunch.types.avro.AvroTypeFamily;
import org.apache.crunch.types.writable.WritableTypeFamily;
import org.apache.mahout.math.Vector;

import java.util.Locale;


/**
 * Specifies the output parameter that can be specified
 *
 * Following commands are support
 * <PRE>
 * <b>--output-type</b></br>
 *      Specifies the output format. Possible values are avro and seq (for SequenceFile)
 *
 * </PRE>
 */
public class OutputParameters {

  public static final String FORMAT_AVRO = "avro";
  public static final String FORMAT_SEQ = "seq";
    public static final String FORMAT_TEXT = "text";

  @Parameter(names = "--output-type", required=true,
      description = "One of 'avro' or 'seq', for Avro or SequenceFile output files")
  private String outputType;

  /**
   * Returns the PType based on Output format specified
   *
   * @return    an instance of {@code PType<T>} </br>
   *            For Avro output format, returns an instance of {@code PType<Vector>} </br>
   *            For SequenceFile output format, returns an instance of {@code AvroType<Vector>}
   */
  public PType<Vector> getVectorPType() {
    outputType = outputType.toLowerCase(Locale.ENGLISH);
    if (FORMAT_AVRO.equals(outputType)) {
      return MLAvros.vector();
    } else if (FORMAT_SEQ.equals(outputType)) {
      return MLWritables.vector();
    } else {
      throw new CommandException("Unsupported Vector output type: " + outputType);
    }
  }
  
  public <T> void write(PCollection<T> collect, String output) {
    outputType = outputType.toLowerCase(Locale.ENGLISH);
    PTypeFamily ptf = collect.getTypeFamily();
    PType<T> ptype = collect.getPType();
    Target target;
    if (FORMAT_TEXT.equals(outputType)) {
      PCollection<String> text = collect.parallelDo(new StringifyFn<T>(),
          collect.getTypeFamily().strings());
      target = To.textFile(output);
      text.write(target, WriteMode.OVERWRITE);
    } else if (FORMAT_AVRO.equals(outputType)) {
      if (AvroTypeFamily.getInstance() != ptf) {
        // Attempt to force conversion
        ptype = AvroTypeFamily.getInstance().as(ptype);
        if (ptype == null) {
          forceConversionException(output, ptype, FORMAT_AVRO);
        }
        collect = collect.parallelDo(IdentityFn.<T>getInstance(), ptype);
      }
      target = At.avroFile(output, (AvroType<T>) ptype);
      collect.write(target, WriteMode.OVERWRITE);
    } else if (FORMAT_SEQ.equals(outputType)) {
          if (WritableTypeFamily.getInstance() != ptf) {
        ptype = WritableTypeFamily.getInstance().as(ptype);
        if (ptype == null) {
          forceConversionException(output, ptype, FORMAT_SEQ);
        }
        collect = collect.parallelDo(IdentityFn.<T>getInstance(), ptype);
      }
      target = At.sequenceFile(output, ptype);
      collect.write(target, WriteMode.OVERWRITE);
    } else {
      throw new CommandException("Unknown output type: " + outputType);
    }
    
  }
  
  private static void forceConversionException(String outputFile, PType<?> ptype, String type) {
    String msg = String.format(
        "Could not convert type %s into %s format for output: %s",
        ptype.getTypeClass().getCanonicalName(), type, outputFile);
    throw new CommandException(msg);
  }
}
