/** Serializing a request replaces unpaired surrogate halves that the strict gateway reader rejects. */
import { describe, expect, it } from 'vitest'
import type { DeepSeekLlmApiExtensionRequest, PreparedDeepSeekLlmApiExtensions } from '@deepseek-ai/dsh-deepseek-llm-api-extensions'
import { prepareRequestExtensions } from '../src/request-extensions.ts'

const signal = new AbortController().signal

/** A preparation that contributes no extension fields. */
async function prepareNothing(): Promise<PreparedDeepSeekLlmApiExtensions> {
  return { fields: {}, accept: async () => {} }
}

/** Serialize one request body through the extension path. */
async function serialize(body: DeepSeekLlmApiExtensionRequest['body']): Promise<string> {
  const { payload } = await prepareRequestExtensions(body, { signal }, prepareNothing, () => {}, () => {})
  return payload
}

describe('prepareRequestExtensions serialization', () => {
  it('replaces an unpaired high surrogate with U+FFFD instead of a bare escape', async () => {
    // U+131A1 is two UTF-16 code units; a cut after the high half leaves one unpaired half.
    const cut = 'x\uD80C'
    expect(cut.isWellFormed()).toBe(false)

    const payload = await serialize({ messages: [{ role: 'user', content: [{ type: 'text', text: cut }] }] })

    expect(payload.isWellFormed()).toBe(true)
    expect(payload).toContain('\uFFFD')
    expect(payload).not.toContain('\\ud80c')
  })

  it('replaces every unpaired half in the conversation', async () => {
    const payload = await serialize({
      messages: [
        { role: 'user', content: [{ type: 'text', text: 'tail \uDC00' }] },
        { role: 'assistant', content: [{ type: 'text', text: '\uD800 head' }] },
      ],
    })

    expect(payload.isWellFormed()).toBe(true)
    expect(payload.split('\uFFFD')).toHaveLength(3)
  })

  it('leaves non-string values and well-formed text untouched', async () => {
    const body = {
      messages: [{ role: 'user', content: [{ type: 'text', text: '\u{131A1} ok' }] }],
      max_tokens: 128,
      stream: true,
      metadata: { nested: [null, 1.5] },
    }

    expect(await serialize(body)).toBe(JSON.stringify(body))
  })
})
