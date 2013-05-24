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

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.crunch.Pipeline;
import org.apache.crunch.impl.mem.MemPipeline;
import org.apache.crunch.impl.mr.MRPipeline;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.compress.CompressionCodec;
import org.apache.hadoop.io.compress.SnappyCodec;
import org.apache.hadoop.util.NativeCodeLoader;

import com.beust.jcommander.Parameter;

/**
 *
 * Pipeline parameters to specify which pipeline to use.
 *
 * Following are possible options
 *
 * <PRE>
 * <b>--local</b>
 *   to ask ML to use Local in-memory pipeline implementation
 *   Default is to use MR pipeline
 * </PRE>
 */
public class PipelineParameters {
  
  private static final Log LOG = LogFactory.getLog(PipelineParameters.class);
  
  @Parameter(names = "--local",
      description = "Use the local, in-memory pipeline implementation")
  private boolean inMemory = false;

  @Parameter(names = "--compress",
      description = "Compress MapReduce output data using Snappy")
  private boolean compress = false;
  
  public Pipeline create(Class<?> jarClass, Configuration conf) {
    if (inMemory) {
      return MemPipeline.getInstance();
    }
    if (compress) {
      conf.setBoolean("mapred.output.compress", true);
      conf.set("mapred.output.compression.type", "BLOCK");
      try {
        if (NativeCodeLoader.buildSupportsSnappy()) {
          conf.setClass("mapred.output.compression.codec", SnappyCodec.class, CompressionCodec.class);
        }
      } catch (UnsatisfiedLinkError e) {
        LOG.warn("Snappy compression disabled in this environment, using default codec");
      }
    }
    return new MRPipeline(jarClass, conf);
  }
}
