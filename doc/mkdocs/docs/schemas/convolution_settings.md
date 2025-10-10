<!-- # ConvolutionSettings -->

<!-- **Title:** ConvolutionSettings
 -->

<!-- |                           |             |
| ------------------------- | ----------- |
| **Type**                  | `object`    |
| **Required**              | No          |
| **Additional properties** | Not allowed |
 -->

<!-- **Description:** Settings for the Convolution sync task.
 -->

**Example:**

```json
{
    "kernel_file": "kernels/gaussian.json",
    "divide": false
}
```

| Property                       | Pattern | Type    | Deprecated | Definition | Title/Description                                                                                                                                                                      |
| ------------------------------ | ------- | ------- | ---------- | ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| + [kernel_file](#kernel_file ) | No      | string  | No         | -          | Path to a JSON file containing a 2D float kernel under the \`kernel\` key. Must satisfy:<br />- kernel_file refers to a readable file<br />- kernel array is rectangular and non-empty |
| + [divide](#divide )           | No      | boolean | No         | -          | When true, divides the input tensor by the convolution result after filtering.                                                                                                         |

<!-- ----------------------------------------------------------------------------------------------------------------------------
Generated using [json-schema-for-humans](https://github.com/coveooss/json-schema-for-humans) on 2025-10-10 at 12:08:06 +0200
-->