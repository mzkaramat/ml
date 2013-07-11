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

import org.apache.crunch.MapFn;
import org.apache.crunch.PCollection;
import org.apache.crunch.Pipeline;
import org.apache.crunch.types.PType;
import org.apache.hadoop.conf.Configuration;

import com.beust.jcommander.Parameter;
import com.beust.jcommander.Parameters;
import com.beust.jcommander.ParametersDelegate;
import com.cloudera.science.ml.avro.MLCenters;
import com.cloudera.science.ml.client.params.PipelineParameters;
import com.cloudera.science.ml.client.params.VectorInputParameters;
import com.cloudera.science.ml.client.params.VectorOutputParameters;
import com.cloudera.science.ml.client.util.AvroIO;
import com.cloudera.science.ml.core.vectors.Centers;
import com.cloudera.science.ml.core.vectors.LabeledVector;
import com.cloudera.science.ml.core.vectors.VectorConvert;
import com.cloudera.science.ml.kmeans.parallel.HardClusterFeaturesFn;
import com.cloudera.science.ml.kmeans.parallel.TriangleClusterFeaturesFn;
import com.cloudera.science.ml.parallel.types.MLAvros;

@Parameters(commandDescription = "Creates a dataset based on a dataset's distance from cluster centers")
public class ClusterFeaturesCommand implements Command {

  @Parameter(names = "--centers-file", required=true,
      description = "A local file to containing the cluster centers")
  private String centersInputFile;
  
  @Parameter(names = "--strategy",
      description = "The strategy for performing the feature mapping, either hard or triangle")
  private String strategy = "hard";
  
  @Parameter(names = "--centers-index",
      description = "The index of the centers in the centers file")
  private int centersIndex = 0;
  
  @Parameter(names = "--output-path", required = true,
      description = "The name of the output path, which will overwrite any existing files with that name")
  private String outputPath;
  
  @ParametersDelegate
  private VectorInputParameters inputParams = new VectorInputParameters();

  @ParametersDelegate
  private PipelineParameters pipelineParams = new PipelineParameters();
  
  @ParametersDelegate
  private VectorOutputParameters outputParams = new VectorOutputParameters();
  
  @Override
  public int execute(Configuration conf) throws IOException {
    List<MLCenters> mlCenterses = AvroIO.read(MLCenters.class, new File(centersInputFile));
    
    Centers centers = VectorConvert.toCenters(mlCenterses.get(centersIndex));
    
    MapFn<LabeledVector, LabeledVector> mapFn;
    if (strategy.equalsIgnoreCase("hard")) {
      mapFn = new HardClusterFeaturesFn(centers);
    } else if (strategy.equalsIgnoreCase("triangle")) {
      mapFn = new TriangleClusterFeaturesFn(centers);
    } else {
      throw new IllegalArgumentException("Invalid strategy: " + strategy);
    }

    Pipeline p = pipelineParams.create(FitCommand.class, conf);

    PCollection<LabeledVector> labeledVectors = inputParams.getLabeledVectors(p);
    PType<LabeledVector> ptype = labeledVectors.getPType();
    PCollection<LabeledVector> outVectors = labeledVectors.parallelDo(mapFn, ptype);
    
    outputParams.writeVectors(outVectors, outputPath, MLAvros.labeledVector());

    p.done();
    
    return 0;
  }

  @Override
  public String getDescription() {
    return "Creates a dataset based on a dataset's distance from cluster centers";
  }

}
