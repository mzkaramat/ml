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
import com.cloudera.science.ml.parallel.fn.SvmLightFn;
import com.cloudera.science.ml.parallel.fn.VectorKeyFns;
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
import org.apache.crunch.types.writable.Writables;
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
  public static final String FORMAT_SVMLIGHT = "svmlight";
  
  private static final String KEY_LONG = "long";
  private static final String KEY_INT = "int";
  private static final String KEY_TEXT = "text";
  
  @Parameter(names = "--output-type", required=true,
      description = "The format for the output vectors, either 'avro', 'svmlight', or 'seq'")
  private String outputType;

  @Parameter(names = "--output-key",
      description = "For 'seq' outputs, the type of the id of each vector, one of 'int', 'long', or 'text'")
  private String keyType;
  
  public <V extends Vector> void writeVectors(PCollection<V> vectors, String output, AvroType<V> atype) {
    outputType = outputType.toLowerCase(Locale.ENGLISH);
    if (FORMAT_AVRO.equals(outputType)) {
      if (AvroTypeFamily.getInstance() != vectors.getTypeFamily()) {
        vectors = vectors.parallelDo(IdentityFn.<V>getInstance(), atype);
      }
      vectors.write(At.avroFile(output, atype), WriteMode.OVERWRITE);
    } else if (FORMAT_SEQ.equals(outputType)) {
      PTypeFamily ptf = WritableTypeFamily.getInstance();
      if (ptf != vectors.getTypeFamily()) {
        vectors = vectors.parallelDo(IdentityFn.<V>getInstance(), atype);
      }
      Target t = To.sequenceFile(output);
      if (keyType != null) {
        keyType = keyType.toLowerCase(Locale.ENGLISH);
        if (KEY_LONG.equals(keyType)) {
          vectors.by(VectorKeyFns.<V>longKeyFn(), ptf.longs()).write(t, WriteMode.OVERWRITE);
        } else if (KEY_INT.equals(keyType)) {
          vectors.by(VectorKeyFns.<V>intKeyFn(), ptf.ints()).write(t, WriteMode.OVERWRITE);
        } else if (KEY_TEXT.equals(keyType)) {
          vectors.by(VectorKeyFns.<V>textKeyFn(), ptf.strings()).write(t, WriteMode.OVERWRITE);
        } else {
          throw new CommandException("Unknown key type: " + keyType);
        }
      } else {
        vectors.write(To.sequenceFile(output), WriteMode.OVERWRITE);
      }
    } else if (FORMAT_SVMLIGHT.equals(outputType)) {
      vectors.parallelDo("svmlight", new SvmLightFn<V>(), Writables.strings())
          .write(To.textFile(output), WriteMode.OVERWRITE);
    } else {
      throw new CommandException("Unknown output type: " + outputType);
    }
  }
}
