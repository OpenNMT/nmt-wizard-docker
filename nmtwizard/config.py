"""Functions to manipulate and validate configurations."""

import jsonschema

from nmtwizard import utils


def index_config(config, path):
    """Index a configuration with a path-like string."""
    for section in path.split('/'):
        if isinstance(config, dict):
            if section not in config:
                raise ValueError('Invalid path %s in config' % path)
            config = config[section]
        elif isinstance(config, list):
            try:
                section_index = int(section)
            except ValueError:
                raise ValueError('Expected an array index in path, but got %s instead' % section)
            config = config[section_index]
        else:
            raise ValueError('Paths in config can only represent object and array structures')
    return config

def index_schema(schema, path):
    """Index a JSON schema with a path-like string."""
    for section in path.split('/'):
        if schema['type'] != 'object':
            raise ValueError('Only object types are supported in the schema structure, '
                             'but saw type %s' % schema['type'])
        properties = schema['properties']
        if section not in properties:
            raise ValueError('Invalid path %s in user options' % path)
        schema = properties[section]
    return schema

def validate_inference_options(inference_options, config):
    """Validate the inference options, raising ValueError on error."""
    json_schema = inference_options.get('json_schema')
    if json_schema is None:
        raise ValueError('Missing "json_schema" in "inference_options"')
    jsonschema.Draft7Validator.check_schema(json_schema)
    options = inference_options.get('options')
    if options is None:
        raise ValueError('Missing "options" in "inference_options"')
    validate_mapping(json_schema, options, config)
    return json_schema

def validate_mapping(schema, options, config):
    """Validate the mapping between inference options and configuration fields,
    raising ValueError on error.
    """
    for i, mapping in enumerate(options):
        config_path = mapping.get('config_path')
        if config_path is None:
            raise ValueError('Missing "config_path" in option mapping %d' % i)
        dst_config = index_config(config, config_path)
        if not isinstance(dst_config, dict):
            raise ValueError('Paths in config can only index object structures')
        option_path = mapping.get('option_path')
        if option_path is None:
            raise ValueError('Missing "option_path" in option mapping %d' % i)
        option_schema = index_schema(schema, option_path)

def update_config_with_options(config, options):
    """Update the configuration with incoming inference options. Raises ValueError
    if inference options were not expected or the value is not accepted.
    """
    inference_options = config.get('inference_options')
    if inference_options is None:
        raise ValueError('This model does not expect inference options')
    try:
        jsonschema.validate(options, inference_options['json_schema'])
    except jsonschema.ValidationError as e:
        raise ValueError('Options validation error: %s' % e.message)
    for mapping in inference_options['options']:
        try:
            option_value = index_config(options, mapping['option_path'])
        except ValueError:
            continue  # Option not passed for this request.
        dst_config = index_config(config, mapping['config_path'])
        dst_config['value'] = option_value
