{#
    content is a template and not a macro in md
        because macro parameters are not through context
        when rendering a template from the macro  and it caused
        serious problems when using recursive calls
    mandatory context parameters:
    schema
#}
{# context parameters default values #}
{% set skip_headers = skip_headers or False %}
{% set depth = depth or 0 %}
{# end context parameters #}

{% set keys = schema.keywords %}
{%- if not skip_headers %}

<!-- {% if schema.title and schema.title | length > 0 %}
**Title:** {{ schema.title }}
{% endif %} -->

<!-- {{ schema | md_type_info_table | md_generate_table }} -->

<!-- {% set description = (schema | get_description) %}
{% include "section_description.md" %}
{% endif %} -->

{# Display examples #}
{% set examples = schema.examples %}
{% if examples %}
    {% include "section_examples.md" %}
{% endif %}

{% if schema.should_be_a_link(config) %}
{% elif schema.refers_to -%}
    {%- with schema=schema.refers_to_merged, skip_headers=True, depth=depth -%}
        {% include "content.md" %}
    {% endwith %}
{% else %}
    {# Properties, pattern properties, additional properties #}
    {% if schema.is_object %}
    {{- schema | md_properties_table | md_generate_table -}}
    {% endif %}

{% endif %}
