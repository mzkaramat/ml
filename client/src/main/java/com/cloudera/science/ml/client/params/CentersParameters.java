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
import com.beust.jcommander.converters.CommaParameterSplitter;
import com.beust.jcommander.converters.IntegerConverter;
import com.cloudera.science.ml.avro.MLCenters;
import com.cloudera.science.ml.client.util.AvroIO;
import com.cloudera.science.ml.core.vectors.Centers;
import com.cloudera.science.ml.core.vectors.VectorConvert;
import com.google.common.collect.Lists;

import java.io.File;
import java.io.IOException;
import java.util.List;

/**
 * Handles parameters related to reading in centers from an external file.
 */
public class CentersParameters {
  @Parameter(names = "--centers-file", required=true,
      description = "The local Avro file containing the centers to be applied")
  private String centersFile;

  @Parameter(names = "--center-ids",
      description = "A CSV containing the indices of the centers to use for the assignment",
      splitter = CommaParameterSplitter.class,
      converter = IntegerConverter.class)
  private List<Integer> centerIds = Lists.newArrayList();

  public List<Integer> getCenterIds() {
    return centerIds;
  }

  public List<Centers> getCenters() throws IOException {
    List<MLCenters> centers = AvroIO.read(MLCenters.class, new File(centersFile));
    if (!centerIds.isEmpty()) {
      List<MLCenters> filter = Lists.newArrayListWithExpectedSize(centerIds.size());
      for (Integer centerId : centerIds) {
        filter.add(centers.get(centerId));
      }
      centers = filter;
    }
    return Lists.transform(centers, VectorConvert.TO_CENTERS);
  }
}
