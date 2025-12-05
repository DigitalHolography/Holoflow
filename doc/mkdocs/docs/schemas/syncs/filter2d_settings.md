<!-- # Filter2DSettings

- [1. Property `Filter2DSettings > r_inner`](#r_inner)
- [2. Property `Filter2DSettings > r_outer`](#r_outer)
- [3. Property `Filter2DSettings > s_inner`](#s_inner)
- [4. Property `Filter2DSettings > s_outer`](#s_outer)

**Title:** Filter2DSettings

|                           |             |
| ------------------------- | ----------- |
| **Type**                  | `object`    |
| **Required**              | No          |
| **Additional properties** | Not allowed |

**Description:** Settings for the Filter2D sync task. -->

**Example:**

```json
{
    "r_inner": 10,
    "r_outer": 100,
    "s_inner": 5,
    "s_outer": 10
}
```

| Property               | Pattern | Type    | Deprecated | Definition | Title/Description                                                                                       |
| ---------------------- | ------- | ------- | ---------- | ---------- | ------------------------------------------------------------------------------------------------------- |
| + [r_inner](#r_inner ) | No      | integer | No         | -          | Inner radius in pixels (non-negative). Controls the inner boundary of the spatial frequency filter.     |
| + [r_outer](#r_outer ) | No      | integer | No         | -          | Outer radius in pixels (non-negative). Controls the outer boundary of the spatial frequency filter.     |
| + [s_inner](#s_inner ) | No      | integer | No         | -          | Inner slope (smoothness) in pixels (non-negative). Defines the smoothing width around the inner radius. |
| + [s_outer](#s_outer ) | No      | integer | No         | -          | Outer slope (smoothness) in pixels (non-negative). Defines the smoothing width around the outer radius. |

<!-- ## <a name="r_inner"></a>1. Property `Filter2DSettings > r_inner`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | Yes       |

**Description:** Inner radius in pixels (non-negative). Controls the inner boundary of the spatial frequency filter.

| Restrictions |        |
| ------------ | ------ |
| **Minimum**  | &ge; 0 |

## <a name="r_outer"></a>2. Property `Filter2DSettings > r_outer`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | Yes       |

**Description:** Outer radius in pixels (non-negative). Controls the outer boundary of the spatial frequency filter.

| Restrictions |        |
| ------------ | ------ |
| **Minimum**  | &ge; 0 |

## <a name="s_inner"></a>3. Property `Filter2DSettings > s_inner`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | Yes       |

**Description:** Inner slope (smoothness) in pixels (non-negative). Defines the smoothing width around the inner radius.

| Restrictions |        |
| ------------ | ------ |
| **Minimum**  | &ge; 0 |

## <a name="s_outer"></a>4. Property `Filter2DSettings > s_outer`

|              |           |
| ------------ | --------- |
| **Type**     | `integer` |
| **Required** | Yes       |

**Description:** Outer slope (smoothness) in pixels (non-negative). Defines the smoothing width around the outer radius.

| Restrictions |        |
| ------------ | ------ |
| **Minimum**  | &ge; 0 |

----------------------------------------------------------------------------------------------------------------------------
Generated using [json-schema-for-humans](https://github.com/coveooss/json-schema-for-humans) on 2025-11-26 at 15:34:20 +0100 -->
