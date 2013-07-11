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
package com.cloudera.science.ml.client.cmd;

import java.io.File;
import java.io.IOException;
import java.util.List;

import com.cloudera.science.ml.client.params.CentersParameters;
import org.apache.crunch.PCollection;
import org.apache.crunch.Pipeline;
import org.apache.hadoop.conf.Configuration;
import org.apache.mahout.math.NamedVector;

import com.beust.jcommander.Parameter;
import com.beust.jcommander.Parameters;
import com.beust.jcommander.ParametersDelegate;
import com.beust.jcommander.converters.CommaParameterSplitter;
import com.beust.jcommander.converters.IntegerConverter;
import com.cloudera.science.ml.avro.MLCenters;
import com.cloudera.science.ml.client.params.PipelineParameters;
import com.cloudera.science.ml.client.params.RecordOutputParameters;
import com.cloudera.science.ml.client.params.VectorInputParameters;
import com.cloudera.science.ml.client.util.AvroIO;
import com.cloudera.science.ml.core.vectors.VectorConvert;
import com.cloudera.science.ml.kmeans.parallel.KMeansParallel;
import com.cloudera.science.ml.parallel.records.Records;
import com.google.common.collect.Lists;

@Parameters(commandDescription =
    "Apply a set of centers to a dataset and output the resulting assignments/distances")
public class KMeansAssignmentCommand implements Command {

  @Parameter(names = "--output-path", required=true,
      description = "The path to write the output to (id, clustering_id, center_id, distance)")
  private String assignmentsPath;
  
  @ParametersDelegate
  private VectorInputParameters inputParams = new VectorInputParameters();
  
  @ParametersDelegate
  private PipelineParameters pipelineParams = new PipelineParameters();

  @ParametersDelegate
  private RecordOutputParameters outputParams = new RecordOutputParameters();

  @ParametersDelegate
  private CentersParameters centersParams = new CentersParameters();

  @Override
  public int execute(Configuration conf) throws IOException {
    Pipeline p = pipelineParams.create(KMeansAssignmentCommand.class, conf);
    PCollection<NamedVector> input = inputParams.getVectors(p);
    KMeansParallel kmp = new KMeansParallel();

    Records assigned = kmp.computeClusterAssignments(input,
        centersParams.getCenters(), centersParams.getCenterIds());

    outputParams.writeRecords(assigned.get(), assigned.getSpec(), assignmentsPath);
    p.done();
    return 0;
  }

  @Override
  public String getDescription() {
    return "Apply a set of centers to a dataset and output the resulting assignments/distances";
  }

}
