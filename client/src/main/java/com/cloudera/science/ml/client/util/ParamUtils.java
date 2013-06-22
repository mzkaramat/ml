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

import java.util.ArrayList;
import java.util.List;

import com.cloudera.science.ml.classifier.core.EtaUpdate;
import com.cloudera.science.ml.classifier.core.OnlineLearnerParams;
import com.cloudera.science.ml.classifier.simple.LinRegOnlineLearner;
import com.cloudera.science.ml.classifier.simple.LogRegOnlineLearner;
import com.cloudera.science.ml.classifier.simple.SVMOnlineLearner;
import com.cloudera.science.ml.classifier.simple.SimpleOnlineLearner;

public class ParamUtils {
  private static final String INT_REGEX = "\\d+";
  private static final String FLOAT_REGEX = "([.]\\d+|\\d+([.]\\d+)?)";
  private static final String FLOAT_RANGE_REGEX = FLOAT_REGEX + "-" + FLOAT_REGEX;
  
  public static List<SimpleOnlineLearner> makeLearners(
      OnlineLearnerParams params, String learnerTypes) {
    String[] tokens = learnerTypes.split("\\s*,\\s*");
    List<SimpleOnlineLearner> learners =
        new ArrayList<SimpleOnlineLearner>(tokens.length);
    for (String token : tokens) {
      learners.add(makeLearner(params, token));
    }
    return learners;
  }
  
  private static SimpleOnlineLearner makeLearner(OnlineLearnerParams params,
      String learnerType) {
    if (learnerType.equalsIgnoreCase("logreg")) {
      return new LogRegOnlineLearner(params);
    } else if (learnerType.equalsIgnoreCase("linreg")) {
      return new LinRegOnlineLearner(params);
    } else if (learnerType.equalsIgnoreCase("svm")) {
      return new SVMOnlineLearner(params);
    } else {
      throw new IllegalArgumentException("Invalid learner type: " + learnerType);
    }
  }
  
  public static EtaUpdate[] parseEtaUpdates(String etaUpdates) {
    String[] tokens = etaUpdates.split("\\s*,\\s*");
    EtaUpdate[] vals = new EtaUpdate[tokens.length];
    for (int i = 0; i < tokens.length; i++) {
      vals[i] = parseEtaUpdate(tokens[i]);
    }
    return vals;
  }
  
  public static EtaUpdate parseEtaUpdate(String etaUpdate) {
    if (etaUpdate.equalsIgnoreCase("CONSTANT")) {
      return EtaUpdate.CONSTANT;
    } else if (etaUpdate.equalsIgnoreCase("BASIC")) {
      return EtaUpdate.BASIC_ETA;
    } else if (etaUpdate.equalsIgnoreCase("PEGASOS")) {
      return EtaUpdate.PEGASOS_ETA;
    } else {
      throw new IllegalArgumentException("Invalid eta update: " + etaUpdate);
    }
  }
  
  public static float[] parseMultivaluedParameter(String param, float bottom,
      float top, int numValues, ParameterInterpolation interpolation) {
    String[] tokens = param.split(",");
    if (tokens.length == 0 || tokens.length > 3) {
      throw new IllegalArgumentException("Illegal parameter value: " + param);
    }
    
    for (String token : tokens) {
      if (token.matches(FLOAT_RANGE_REGEX)) {
        int dashIndex = tokens[0].indexOf('-');
        String bottomStr = tokens[0].substring(0, dashIndex);
        bottom = Float.parseFloat(bottomStr);
        String topStr = tokens[0].substring(dashIndex+1);
        top = Float.parseFloat(topStr);
      } else if (token.matches(INT_REGEX)) {
        numValues = Integer.parseInt(token);
      } else if (token.equalsIgnoreCase("lin")) {
        interpolation = ParameterInterpolation.LINEAR;
      } else if (token.endsWith("exp")) {
        interpolation = ParameterInterpolation.EXPONENTIAL;
      } else {
        throw new IllegalArgumentException("Illegal parameter value: " + param);
      }
    }
    
    float[] vals = new float[numValues];
    if (interpolation == ParameterInterpolation.LINEAR) {
      interpolateLinear(bottom, top, vals);
    } else if (interpolation == ParameterInterpolation.EXPONENTIAL) {
      interpolateExponential(bottom, top, vals);
    }
    
    return vals;
  }
  
  static void interpolateLinear(float bottom, float top, float[] vals) {
    double multiplier = (top - bottom) / (vals.length-1);
    double start = bottom / multiplier;
    for (int i = 0; i < vals.length; i++) {
      vals[i] = (float)(multiplier * (start + i));
    }
  }
  
  static void interpolateExponential(float bottom, float top, float[] vals) {
    double base = Math.pow(top / bottom, 1.0 / (vals.length-1));
    double startPower = Math.log10(bottom) / Math.log10(base);
    for (int i = 0; i < vals.length; i++) {
      vals[i] = (float)Math.pow(base, startPower + i);
    }
  }
}
