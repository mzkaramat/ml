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

import java.util.Collections;
import java.util.List;
import java.util.Locale;

import org.apache.crunch.PCollection;
import org.apache.crunch.Pipeline;
import org.apache.crunch.io.From;
import org.apache.mahout.math.Vector;

import com.beust.jcommander.Parameter;
import com.beust.jcommander.converters.CommaParameterSplitter;
import com.cloudera.science.ml.client.cmd.CommandException;
import com.cloudera.science.ml.client.util.UnionIO;
import com.cloudera.science.ml.mahout.types.MLWritables;
import com.cloudera.science.ml.parallel.types.MLAvros;
import com.google.common.base.Function;

/**
 * Class specifies the common input parameters that may be used across ML commands
 * that process vectors.
 *
 * The following options are supported:
 *
 * <PRE>
 * <b>--input-paths</b>
 *     Comma separated paths to be used as input
 *
 * <b>--format</b>
 *     format of the Input. Possible values are seq and avro
 */
public class VectorInputParameters {

  public static final String FORMAT_SEQ = "seq";
  public static final String FORMAT_AVRO = "avro";
  
  @Parameter(names = "--input-paths",
      description = "CSV of the input paths to consider",
      splitter = CommaParameterSplitter.class,
      required = true)
  private List<String> inputPaths;

  @Parameter(names = "--format",
      description = "Either 'seq' or 'avro' to describe the format of the input vectors",
      required = true)
  private String format;
  
  public <V extends Vector> PCollection<V> getVectorsFromPath(Pipeline pipeline, String path) {
    return (PCollection<V>) getVectors(pipeline, Collections.singletonList(path));
  }
  
  public <V extends Vector> PCollection<V> getVectors(Pipeline pipeline) {
    return (PCollection<V>) getVectors(pipeline, inputPaths);
  }
  
  private PCollection<Vector> getVectors(final Pipeline pipeline, List<String> paths) {
    format = format.toLowerCase(Locale.ENGLISH);
    PCollection<Vector> ret;
    if (FORMAT_SEQ.equals(format)) {
      ret = UnionIO.from(paths, new Function<String, PCollection<Vector>>() {
        @Override
        public PCollection<Vector> apply(String input) {
          return pipeline.read(From.sequenceFile(input, MLWritables.vector()));
        }
      });
    } else if (FORMAT_AVRO.equals(format)) {
      ret = UnionIO.from(paths, new Function<String, PCollection<Vector>>() {
        @Override
        public PCollection<Vector> apply(String input) {
          return pipeline.read(From.avroFile(input, MLAvros.vector()));
        }
      });
    } else {
      throw new CommandException("Unsupported vector format: " + format);
    }
    return ret;
  }

}
