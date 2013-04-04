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
package com.cloudera.science.ml.core.records;

import java.io.File;
import java.io.IOException;
import java.util.Collection;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.regex.Pattern;

import com.google.common.base.Charsets;
import com.google.common.base.Function;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Lists;
import com.google.common.io.Files;

/**
 * Utility functions for working with header files and {@code Spec} data.
 */
public final class Specs {

  // Indicators that a header file contains metadata
  private static final Set<String> SYMBOL_META = ImmutableSet.of("symbolic",
      "categorical", "nominal", "string");
  private static final Set<String> NUMERIC_META = ImmutableSet.of("numeric",
      "continuous", "real", "double");
  private static final Pattern COMMA = Pattern.compile(",");

  public static Spec readFromHeaderFile(String headerFile) throws IOException {
    return readFromHeaderFile(headerFile, null, null);
  }
  
  public static Spec readFromHeaderFile(String headerFile,
                                        Collection<Integer> ignoredColumns,
                                        Collection<Integer> symbolicColumns) throws IOException {
    List<String> lines = Files.readLines(new File(headerFile), Charsets.UTF_8);
    RecordSpec.Builder rsb = RecordSpec.builder();
    for (int i = 0; i < lines.size(); i++) {
      String line = lines.get(i);
      if (line.contains(",")) {
        String[] pieces = COMMA.split(line);
        if (pieces.length != 2) {
          throw new IllegalArgumentException("Invalid header file row: " + line);
        }
        String name = pieces[0];
        String meta = pieces[1].toLowerCase(Locale.ENGLISH).trim();
        if (meta.startsWith("ignore") || meta.startsWith("id")) {
          if (ignoredColumns != null) {
            ignoredColumns.add(i);
          }
          rsb.add(name, DataType.STRING);
        } else if (SYMBOL_META.contains(meta)) {
          if (symbolicColumns != null) {
            symbolicColumns.add(i);
          }
          rsb.add(name, DataType.STRING);
        } else if (NUMERIC_META.contains(meta)) {
          rsb.add(name, DataType.DOUBLE);
        } else {
          throw new IllegalArgumentException(String.format(
              "Did not recognize metadata %s for field %s", meta, name));
        }
      } else {
        rsb.add(line, DataType.DOUBLE);
      }
    }
    return rsb.build();  
  }
  
  public static boolean isNumeric(Spec spec, String fieldId) {
    FieldSpec fs = spec.getField(getFieldId(spec, fieldId));
    return fs.spec().getDataType().isNumeric();
  }
  
  public static Integer getFieldId(Spec spec, String value) {
    List<Integer> fieldIds = getFieldIds(spec, ImmutableList.of(value));
    if (fieldIds.isEmpty()) {
      throw new IllegalArgumentException("Could not find field " + value + " in spec");
    }
    return fieldIds.get(0);
  }
  
  public static List<Integer> getFieldIds(Spec spec, List<String> values) {
    if (values.isEmpty()) {
      return ImmutableList.of();
    }
    
    List<Integer> fieldIds;
    if (spec == null || spec.getField(values.get(0)) == null) {
      fieldIds = Lists.transform(values, new Function<String, Integer>() {
        @Override
        public Integer apply(String input) {
          try {
            return Integer.valueOf(input);
          } catch (NumberFormatException ignored) {
            throw new IllegalArgumentException("Did not recognize column ID: " + input);
          }
        }
      });
    } else {
      fieldIds = Lists.newArrayListWithExpectedSize(values.size());
      for (String value : values) {
        FieldSpec f = spec.getField(value);
        if (f != null) {
          fieldIds.add(f.position());
        }
      }
    }
    return fieldIds;
  }
  
  private Specs() {
  }
}
