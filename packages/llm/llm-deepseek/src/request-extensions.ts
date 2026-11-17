/** Prepare plugin-contributed request fields and commit their delivery after HTTP acceptance. */

import { errorChain, LlmError } from '@deepseek-ai/dsh-llm'
import type { DeepSeekLlmApiExtensionRequest, PreparedDeepSeekLlmApiExtensions } from '@deepseek-ai/dsh-deepseek-llm-api-extensions'
import type { DeepSeekAdapterOptions } from './types.ts'

/** An unpaired UTF-16 surrogate half: a high half with no low half after it, or a low half with none before. */
const LONE_SURROGATE = /[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/g

/**
 * Replace unpaired surrogate halves with U+FFFD while serializing.
 * Text reaching a request can carry a lone surrogate — a UTF-16 slice that cut an
 * astral character in half, for example. `JSON.stringify` emits it as a bare
 * `\uD8xx` escape, which strict parsers (Rust/serde_json behind the gateway)
 * reject as an unexpected end of hex escape, failing the whole request with 400.
 * @param _key - replacer key, unused.
 * @param value - the value `JSON.stringify` is about to encode.
 * @returns the value with any unpaired surrogate half replaced.
 */
function replaceLoneSurrogates(_key: string, value: unknown): unknown {
  return typeof value === 'string' ? value.replace(LONE_SURROGATE, '\uFFFD') : value
}

/**
 * Merge contributions without replacing Messages fields. A rejected preparation or
 * a base-field collision reports REQUEST_EXTENSION with the cause chain appended to
 * the message. When the merged request fails to serialize, the payload is the
 * base request alone and acceptance is a no-op, so contributors resend their
 * unaccepted state on a later request. An acceptance failure is reported and
 * does not fail the request.
 * @param body - serialized Messages request before extension fields.
 * @param options - request identity, purpose, and cancellation.
 * @param prepare - contributor registry captured for this adapter.
 * @param onOmitted - receives the omitted field names and the serialization failure.
 * @param onUnaccepted - receives an acceptance failure after HTTP 2xx.
 * @returns HTTP payload and a commit to invoke only after a successful HTTP response.
 */
export async function prepareRequestExtensions(
  body: DeepSeekLlmApiExtensionRequest['body'],
  options: Omit<DeepSeekLlmApiExtensionRequest, 'body'>,
  prepare: DeepSeekAdapterOptions['prepareExtensions'],
  onOmitted: (fields: readonly string[], error: unknown) => void,
  onUnaccepted: (error: unknown) => void,
): Promise<{ payload: string; accept(): Promise<void> }> {
  let extensions: PreparedDeepSeekLlmApiExtensions
  try {
    extensions = await prepare({ body, ...options })
  } catch (error) {
    throw new LlmError(`DeepSeek request extension preparation failed: ${errorChain(error)}`, 'REQUEST_EXTENSION', { cause: error })
  }
  const fields = Object.keys(extensions.fields)
  for (const field of fields) {
    if (Object.hasOwn(body, field)) {
      throw new LlmError(`DeepSeek request extension field ${JSON.stringify(field)} collides with the base request`, 'REQUEST_EXTENSION')
    }
  }
  let payload: string
  try {
    payload = JSON.stringify({ ...body, ...extensions.fields }, replaceLoneSurrogates)
  } catch (error) {
    // A base request that cannot serialize throws here, before any field is reported as omitted.
    const base = JSON.stringify(body, replaceLoneSurrogates)
    onOmitted(fields, error)
    return { payload: base, accept: () => Promise.resolve() }
  }
  return {
    payload,
    async accept() {
      try {
        await extensions.accept()
      } catch (error) {
        onUnaccepted(error)
      }
    },
  }
}
