/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements. See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

// Report the diagram's natural height and receive the documentation theme.
(() => {
  const root = document.documentElement;
  const dark = matchMedia('(prefers-color-scheme: dark)');
  root.dataset.theme = dark.matches ? 'dark' : 'light';
  if (parent === window) {
    dark.addEventListener('change', () => { root.dataset.theme = dark.matches ? 'dark' : 'light'; });
    return;
  }
  const parentOrigin = document.referrer ? new URL(document.referrer).origin : '*';
  const report = () => parent.postMessage({
    type: 'tirx-matrix-dataflow:height', height: Math.ceil(document.body.getBoundingClientRect().height)
  }, parentOrigin);
  new ResizeObserver(report).observe(document.body);
  addEventListener('message', (event) => {
    if (event.source !== parent || (parentOrigin !== '*' && event.origin !== parentOrigin)) return;
    if (event.data?.type !== 'tirx-matrix-dataflow:theme') return;
    if (['light', 'dark'].includes(event.data.theme)) root.dataset.theme = event.data.theme;
    report();
  });
  addEventListener('load', report);
  report();
})();
