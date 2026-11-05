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

// Keep the hosted diagram's height and theme in sync with its documentation page.
(() => {
  const dark = matchMedia('(prefers-color-scheme: dark)');
  document.querySelectorAll('iframe[data-tirx-authoring-layers]').forEach((frame) => {
    if (frame.dataset.ready) return;
    frame.dataset.ready = 'true';
    const origin = new URL(frame.src, document.baseURI).origin;
    const syncTheme = () => {
      const selected = document.documentElement.dataset.theme;
      const theme = ['light', 'dark'].includes(selected) ? selected : (dark.matches ? 'dark' : 'light');
      frame.contentWindow.postMessage({ type: 'tirx-authoring-layers:theme', theme }, origin);
    };
    addEventListener('message', (event) => {
      if (event.source !== frame.contentWindow || event.origin !== origin) return;
      const data = event.data;
      if (data?.type !== 'tirx-authoring-layers:height') return;
      if (!Number.isFinite(data.height) || data.height < 100 || data.height > 2000) return;
      frame.style.height = Math.ceil(data.height) + 'px';
    });
    new MutationObserver(syncTheme).observe(document.documentElement, {
      attributes: true, attributeFilter: ['data-theme']
    });
    dark.addEventListener('change', syncTheme);
    frame.addEventListener('load', syncTheme);
    syncTheme();
  });
})();
