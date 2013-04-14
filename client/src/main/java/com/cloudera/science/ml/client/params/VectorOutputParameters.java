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
import com.cloudera.science.ml.parallel.types.MLAvros;
import org.apache.crunch.PCollection;
import org.apache.crunch.Target.WriteMode;
import org.apache.crunch.fn.IdentityFn;
import org.apache.crunch.io.At;
import org.apache.crunch.io.To;
import org.apache.crunch.types.PType;
import org.apache.crunch.types.avro.AvroType;
import org.apache.crunch.types.avro.AvroTypeFamily;
import org.apache.crunch.types.writable.WritableTypeFamily;
import org.apache.mahout.math.Vector;

import java.util.Locale;


/**
 * Specifies the formats that can be used with vector outputs.
 *
 * Following commands are support
 * <PRE>
 * <b>--output-type</b></br>
 *      Specifies the output format. Possible values are avro and seq (for SequenceFile)
 *
 * </PRE>
 */
public class VectorOutputParameters {

  public static final String FORMAT_AVRO = "avro";
  public static final String FORMAT_SEQ = "seq";

  @Parameter(names = "--output-type", required=true,
      description = "The format for the output vectors, either 'avro' or 'seq'")
  private String outputType;

  /**
   * Returns the PType based on the output format specified
   *
   * @return    an instance of {@code PType<Vector>} </br>
   */
  public PType<Vector> getPType() {
    outputType = outputType.toLowerCase(Locale.ENGLISH);
    if (FORMAT_AVRO.equals(outputType)) {
      return MLAvros.vector();
    } else if (FORMAT_SEQ.equals(outputType)) {
      return MLWritables.vector();
    } else {
      throw new CommandException("Unsupported Vector output type: " + outputType);
    }
  }
  
  public <V extends Vector> void writeVectors(PCollection<V> vectors, String output) {
    outputType = outputType.toLowerCase(Locale.ENGLISH);
    if (FORMAT_AVRO.equals(outputType)) {
      AvroType<Vector> atype = MLAvros.vector();
      if (AvroTypeFamily.getInstance() != vectors.getTypeFamily()) {
        vectors = vectors.parallelDo(IdentityFn.<V>getInstance(), (PType<V>) MLAvros.vector());
      }
      vectors.write(At.avroFile(output, atype), WriteMode.OVERWRITE);
    } else if (FORMAT_SEQ.equals(outputType)) {
      if (WritableTypeFamily.getInstance() != vectors.getTypeFamily()) {
        vectors = vectors.parallelDo(IdentityFn.<V>getInstance(), (PType<V>) MLWritables.vector());
      }
      vectors.write(To.sequenceFile(output), WriteMode.OVERWRITE);
    } else {
      throw new CommandException("Unknown output type: " + outputType);
    }
  }
}
