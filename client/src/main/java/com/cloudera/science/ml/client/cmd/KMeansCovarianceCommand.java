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

import com.beust.jcommander.Parameter;
import com.beust.jcommander.Parameters;
import com.beust.jcommander.ParametersDelegate;
import com.beust.jcommander.converters.CommaParameterSplitter;
import com.beust.jcommander.converters.IntegerConverter;
import com.cloudera.science.ml.avro.MLCenters;
import com.cloudera.science.ml.avro.MLClusterCovariance;
import com.cloudera.science.ml.client.params.PipelineParameters;
import com.cloudera.science.ml.client.params.VectorInputParameters;
import com.cloudera.science.ml.client.util.AvroIO;
import com.cloudera.science.ml.core.vectors.VectorConvert;
import com.cloudera.science.ml.kmeans.parallel.KMeansParallel;
import com.google.common.collect.Lists;
import org.apache.crunch.PCollection;
import org.apache.crunch.PObject;
import org.apache.crunch.Pipeline;
import org.apache.hadoop.conf.Configuration;
import org.apache.mahout.math.Vector;

import java.io.File;
import java.io.IOException;
import java.util.List;

@Parameters(commandDescription =
    "Compute the covariance matrices for the centers in one or more clusters")
public class KMeansCovarianceCommand implements Command {

  @Parameter(names = "--centers-file", required=true,
      description = "The local Avro file containing the centers to be applied")
  private String centersFile;

  @Parameter(names = "--center-ids",
      description = "A CSV containing the indices of the centers to use for the assignment",
      splitter = CommaParameterSplitter.class,
      converter = IntegerConverter.class)
  private List<Integer> centerIds = Lists.newArrayList();

  @Parameter(names = "--output-file", required=true,
      description = "The local file to write with the Avro-formatted covariance output")
  private String outputFile;

  @Parameter(names = "--approx",
      description = "Enable approximate center assignments to speed up processing at the cost of accuracy")
  private boolean approx = false;

  @ParametersDelegate
  private VectorInputParameters inputParams = new VectorInputParameters();

  @ParametersDelegate
  private PipelineParameters pipelineParams = new PipelineParameters();

  @Override
  public int execute(Configuration conf) throws IOException {
    Pipeline p = pipelineParams.create(KMeansCovarianceCommand.class, conf);
    PCollection<Vector> vecs = inputParams.getVectors(p);

    List<MLCenters> centers = AvroIO.read(MLCenters.class, new File(centersFile));
    if (!centerIds.isEmpty()) {
      List<MLCenters> filter = Lists.newArrayListWithExpectedSize(centerIds.size());
      for (Integer centerId : centerIds) {
        filter.add(centers.get(centerId));
      }
      centers = filter;
    }
    KMeansParallel kmp = new KMeansParallel();

    PObject<List<MLClusterCovariance>> cov = kmp.computeClusterCovarianceMatrix(vecs,
        Lists.transform(centers, VectorConvert.TO_CENTERS),
        approx,
        centerIds);
    AvroIO.write(cov.getValue(), new File(outputFile));

    p.done();
    return 0;
  }

  @Override
  public String getDescription() {
    return "Compute the covariance matrices for the centers in one or more clusters";
  }
}
