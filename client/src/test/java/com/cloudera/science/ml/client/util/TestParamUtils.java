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

package com.cloudera.science.ml.client.util;

import org.junit.Assert;
import org.junit.Test;

public class TestParamUtils {
  @Test
  public void testInterpolateLinear() {
    float[] vals = new float[4];
    float bottom = .5f;
    float top = 2.0f;
    ParamUtils.interpolateLinear(bottom, top, vals);
    Assert.assertArrayEquals(new float[] {.5f, 1.0f, 1.5f, 2.0f}, vals, .001f);
  }
  
  @Test
  public void testInterpolateExponential() {
    float[] vals = new float[4];
    float bottom = .5f;
    float top = 4.0f;
    ParamUtils.interpolateExponential(bottom, top, vals);
    Assert.assertArrayEquals(new float[] {.5f, 1.0f, 2.0f, 4.0f}, vals, .001f);
  }
}
