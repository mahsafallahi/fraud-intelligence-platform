-- Fails (returns a row) when the model does not have exactly `expected` rows.
{% test row_count_equals(model, expected) %}

select count(*) as actual_rows
from {{ model }}
having count(*) <> {{ expected }}

{% endtest %}
