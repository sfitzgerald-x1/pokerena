const {calculate, Field, Generations, Move, Pokemon} = require('@smogon/calc');

const DAMAGE_SUPPORT_SCHEMA_VERSION = 'pokerena.damage-support.v1';

function fail(message) {
  const error = new Error(message);
  error.name = 'DamageCalcError';
  throw error;
}

function requireObject(value, label) {
  if (!value || Array.isArray(value) || typeof value !== 'object') {
    fail(`${label} must be a JSON object.`);
  }
  return value;
}

function requireString(value, label) {
  if (typeof value !== 'string' || !value.trim()) {
    fail(`${label} must be a non-empty string.`);
  }
  return value;
}

function optionalObject(value, label) {
  if (value === undefined || value === null) {
    return {};
  }
  return requireObject(value, label);
}

function requireSchemaVersion(value) {
  if (value !== 'pokerena.damage-request.v1') {
    fail("schema_version must be 'pokerena.damage-request.v1'.");
  }
}

function normalizeDamage(damage) {
  if (typeof damage === 'number') {
    return damage;
  }
  if (Array.isArray(damage)) {
    return damage.map(normalizeDamage);
  }
  fail('Damage result was not a number or array.');
}

function percent(value, total) {
  if (!total) {
    return 0;
  }
  return Number(((value / total) * 100).toFixed(2));
}

function buildPokemon(gen, value, label) {
  const data = requireObject(value, label);
  const species = requireString(data.species, `${label}.species`);
  const options = optionalObject(data.options, `${label}.options`);
  return new Pokemon(gen, species, options);
}

function buildMove(gen, value) {
  const data = requireObject(value, 'move');
  const name = requireString(data.name, 'move.name');
  const options = optionalObject(data.options, 'move.options');
  return new Move(gen, name, options);
}

function requireGeneration(value) {
  if (!Number.isInteger(value) || value < 1) {
    fail('generation must be a positive integer.');
  }
  const gen = Generations.get(value);
  if (!gen) {
    fail(`Unsupported generation: ${value}`);
  }
  return gen;
}

function classifierPayload(payload) {
  const data = requireObject(payload, 'payload');
  const generation = data.generation;
  const gen = requireGeneration(generation);
  const move = buildMove(gen, data.move);
  return {data, generation, gen, move};
}

function classifyMoveSupport(payload) {
  const {generation, gen, move} = classifierPayload(payload);
  const moveName = move.name;
  if (move.category === 'Status') {
    return {
      schema_version: DAMAGE_SUPPORT_SCHEMA_VERSION,
      generation,
      move_name: moveName,
      classification: 'supported_non_damaging',
      reason: 'status-category',
    };
  }
  if (typeof move.type !== 'string' || !move.type) {
    return {
      schema_version: DAMAGE_SUPPORT_SCHEMA_VERSION,
      generation,
      move_name: moveName,
      classification: 'unsupported',
      reason: 'missing-move-type',
    };
  }

  try {
    const attacker = new Pokemon(gen, 'Mew', {level: 100});
    const defender = new Pokemon(gen, 'Mew', {level: 100});
    const result = calculate(gen, attacker, defender, move, new Field({}));
    const range = result.range();
    const maxRange = Array.isArray(range) ? range[1] : 0;
    const damage = normalizeDamage(result.damage);
    const maxDamage = Array.isArray(damage) ? Math.max(...damage) : damage;
    if (maxRange > 0 || maxDamage > 0) {
      return {
        schema_version: DAMAGE_SUPPORT_SCHEMA_VERSION,
        generation,
        move_name: moveName,
        classification: 'supported_damaging',
        reason: 'calc-ok',
      };
    }
  } catch (error) {
    return {
      schema_version: DAMAGE_SUPPORT_SCHEMA_VERSION,
      generation,
      move_name: moveName,
      classification: 'unsupported',
      reason: error instanceof Error ? error.message : String(error),
    };
  }

  return {
    schema_version: DAMAGE_SUPPORT_SCHEMA_VERSION,
    generation,
    move_name: moveName,
    classification: 'supported_non_damaging',
    reason: 'zero-damage-result',
  };
}

function calculateDamageFromParsed(data, gen, move) {
  const attacker = buildPokemon(gen, data.attacker, 'attacker');
  const defender = buildPokemon(gen, data.defender, 'defender');
  const field = new Field(optionalObject(data.field, 'field'));
  const result = calculate(gen, attacker, defender, move, field);
  const range = result.range();
  const defenderHP = defender.maxHP();
  const kochance = result.kochance();

  return {
    schema_version: 'pokerena.damage-result.v1',
    generation: gen.num,
    attacker: {
      species: attacker.name,
      level: attacker.level,
    },
    defender: {
      species: defender.name,
      level: defender.level,
      hp: defenderHP,
    },
    move: {
      name: move.name,
    },
    damage: normalizeDamage(result.damage),
    range: {
      min: range[0],
      max: range[1],
    },
    range_percent: {
      min: percent(range[0], defenderHP),
      max: percent(range[1], defenderHP),
    },
    description: result.desc(),
    knockout: {
      chance: kochance.chance ?? null,
      hits: kochance.n,
      text: kochance.text,
    },
  };
}

function calculateDamage(payload) {
  const {data, generation, gen, move} = classifierPayload(payload);
  const support = classifyMoveSupport(payload);
  if (support.classification === 'supported_non_damaging') {
    fail(`Damage calc is not applicable for non-damaging move ${move.name} in gen ${generation}.`);
  }
  if (support.classification === 'unsupported') {
    fail(`Damage calc is unsupported for move ${move.name} in gen ${generation}.`);
  }
  return calculateDamageFromParsed(data, gen, move);
}

function calculateDamageBatch(payload) {
  const data = requireObject(payload, 'payload');
  if (data.schema_version !== 'pokerena.damage-batch-request.v1') {
    fail("schema_version must be 'pokerena.damage-batch-request.v1'.");
  }
  if (!Array.isArray(data.requests) || data.requests.length < 1) {
    fail('requests must be a non-empty array.');
  }
  return {
    schema_version: 'pokerena.damage-batch-result.v1',
    results: data.requests.map(requestPayload => {
      const support = classifyMoveSupport(requestPayload);
      if (support.classification === 'supported_damaging') {
        return {
          status: 'ok',
          move_name: support.move_name,
          generation: support.generation,
          result: calculateDamage(requestPayload),
        };
      }
      return {
        status: 'skipped',
        skip_reason:
          support.classification === 'supported_non_damaging' ? 'non_damaging' : 'unsupported',
        move_name: support.move_name,
        generation: support.generation,
      };
    }),
  };
}

module.exports = {
  calculateDamage,
  calculateDamageBatch,
  classifyMoveSupport,
  fail,
};
