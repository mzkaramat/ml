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
import com.cloudera.science.ml.client.params.VectorInputParameters;
import com.cloudera.science.ml.client.util.AvroIO;
import com.cloudera.science.ml.core.vectors.Centers;
import com.cloudera.science.ml.core.vectors.VectorConvert;
import com.cloudera.science.ml.kmeans.parallel.KMeansParallel;
import com.google.common.collect.Lists;

@Parameters(commandDescription =
    "Run one or more iterations of Lloyd's algorithm over data in HDFS to optimize the output of kmeans")
public class LloydsCommand implements Command {

  @Parameter(names = "--centers-file", required=true,
      description = "The local Avro file containing the centers to be optimized")
  private String centersFile;
  
  @Parameter(names = "--center-ids",
      description = "A CSV containing the indices of the centers to update. If not specified, all centers will be updated",
      splitter = CommaParameterSplitter.class,
      converter = IntegerConverter.class)
  private List<Integer> centerIds = Lists.newArrayList();
  
  @Parameter(names = "--output-centers-file", required=true,
      description = "The local Avro file to write the optimizer centers to")
  private String outputCentersFile;
  
  @Parameter(names = "--num-iterations",
      description = "The number of iterations of Lloyd's algorithm to run")
  private int numIterations = 1;
  
  @Parameter(names = "--approx",
      description = "Use approximate point assignment (tends to speed up runs at the cost of accuracy)")
  private boolean approx = false;
  
  @Parameter(names = "--compute-costs",
      description = "If true, calculates and prints the cost of the new output clusters")
  private boolean computeCosts = false;
  
  @ParametersDelegate
  private VectorInputParameters inputParams = new VectorInputParameters();
  
  @ParametersDelegate
  private PipelineParameters pipelineParams = new PipelineParameters();
  
  @Override
  public int execute(Configuration conf) throws IOException {
    Pipeline p = pipelineParams.create(KMeansAssignmentCommand.class, conf);
    PCollection<NamedVector> input = inputParams.getVectors(p);
    List<MLCenters> mlCenters = AvroIO.read(MLCenters.class, new File(centersFile));
    if (!centerIds.isEmpty()) {
      List<MLCenters> filter = Lists.newArrayListWithExpectedSize(centerIds.size());
      for (Integer centerId : centerIds) {
        filter.add(mlCenters.get(centerId));
      }
      mlCenters = filter;
    }
    
    KMeansParallel kmp = new KMeansParallel();
    List<Centers> initial = Lists.transform(mlCenters, VectorConvert.TO_CENTERS);
    List<Centers> output = kmp.lloydsAlgorithm(input, initial, numIterations, approx);
    if (computeCosts) {
      List<Double> costs = kmp.getCosts(input, output, approx).getValue();
      System.out.println("CenterId,Cost");
      for (int i = 0; i < costs.size(); i++) {
        int centerId = centerIds.isEmpty() ? i : centerIds.get(i);
        System.out.println(String.format("%d,%.4f", centerId, costs.get(i)));
      }
    }
    
    AvroIO.write(Lists.transform(output, VectorConvert.FROM_CENTERS), new File(outputCentersFile));
    p.done();
    return 0;
  }

  @Override
  public String getDescription() {
    return "Run one or more iterations of Lloyd's algorithm over data in HDFS to optimize the output of kmeans";
  }

}
