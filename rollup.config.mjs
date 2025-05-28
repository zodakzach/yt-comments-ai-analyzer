import terser from "@rollup/plugin-terser";
import { nodeResolve } from "@rollup/plugin-node-resolve";

export default {
  input: "static/js/index.js",
  output: {
    file: "static/js/bundle.js",
    format: "iife",
    sourcemap: true,
  },
  plugins: [
    nodeResolve(), // 👈 This resolves htmx.org in node_modules
    terser(),
  ],
};
