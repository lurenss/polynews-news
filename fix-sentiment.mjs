#!/usr/bin/env node
/**
 * Fix Sentiment Script
 *
 * Re-analyzes sentiment for all news articles using OpenAI GPT-5-nano.
 * The original code had insufficient max_completion_tokens for the reasoning model.
 *
 * Usage:
 *   OPENAI_API_KEY=sk-xxx node fix-sentiment.mjs
 *
 * Options:
 *   --dry-run    Preview changes without saving
 *   --limit N    Process only N markets (for testing)
 */

import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Configuration
const OPENAI_API_URL = 'https://api.openai.com/v1/chat/completions';
const OPENAI_MODEL = 'gpt-5-nano';
const NEWS_DIR = path.join(__dirname, 'news');
const TOP_NEWS_FILE = path.join(__dirname, 'top-news.json');
const BATCH_SIZE = 100; // Markets to process concurrently (high RPM available)
const RETRY_DELAY_MS = 1000;
const MAX_RETRIES = 3;
const MAX_COMPLETION_TOKENS = 1500; // gpt-5-nano needs lots of tokens for reasoning
const ARTICLES_PER_BATCH = 5; // Process articles in smaller batches to limit prompt size

// Parse command line args
const args = process.argv.slice(2);
const DRY_RUN = args.includes('--dry-run');
const limitIndex = args.indexOf('--limit');
const LIMIT = limitIndex !== -1 ? parseInt(args[limitIndex + 1], 10) : null;

// Stats tracking
const stats = {
  filesProcessed: 0,
  articlesProcessed: 0,
  sentimentChanges: { bullish: 0, bearish: 0, neutral: 0 },
  errors: 0,
};

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/**
 * Call OpenAI API with retry logic
 */
async function callOpenAI(messages) {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) {
    throw new Error('OPENAI_API_KEY environment variable is required');
  }

  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    try {
      const response = await fetch(OPENAI_API_URL, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${apiKey}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: OPENAI_MODEL,
          messages,
          max_completion_tokens: MAX_COMPLETION_TOKENS,
        }),
      });

      if (!response.ok) {
        const errorBody = await response.text();
        if ([429, 502, 503, 504].includes(response.status) && attempt < MAX_RETRIES) {
          const delay = RETRY_DELAY_MS * Math.pow(2, attempt);
          console.warn(`  ⚠️  Rate limited (${response.status}), retrying in ${delay}ms...`);
          await sleep(delay);
          continue;
        }
        throw new Error(`OpenAI API error: ${response.status} - ${errorBody}`);
      }

      const data = await response.json();
      return data.choices?.[0]?.message?.content?.trim() || '';
    } catch (error) {
      if (attempt < MAX_RETRIES && !error.message.startsWith('OpenAI API error:')) {
        const delay = RETRY_DELAY_MS * Math.pow(2, attempt);
        console.warn(`  ⚠️  Request failed, retrying in ${delay}ms...`);
        await sleep(delay);
        continue;
      }
      throw error;
    }
  }
}

/**
 * Analyze sentiment of a small batch of articles
 */
async function analyzeSmallBatch(articles, marketTitle) {
  // Truncate summaries to limit prompt size
  const articleList = articles.map((a, i) => {
    const summary = (a.summary || '').slice(0, 100);
    return `${i + 1}. ${a.title.slice(0, 80)}: ${summary}`;
  }).join('\n');

  try {
    const result = await callOpenAI([
      {
        role: 'system',
        content: 'Analyze if each news article is BULLISH (suggests YES), BEARISH (suggests NO), or NEUTRAL for the prediction market. Return ONLY a JSON array of sentiments.',
      },
      {
        role: 'user',
        content: `Market: "${marketTitle.slice(0, 100)}"

Articles:
${articleList}

Return JSON array: ["bullish", "neutral", "bearish", ...]`,
      },
    ]);

    const match = result.match(/\[.*\]/s);
    if (match) {
      const parsed = JSON.parse(match[0]);
      return parsed.map((s) => {
        const lower = String(s).toLowerCase();
        if (lower === 'bullish') return 'bullish';
        if (lower === 'bearish') return 'bearish';
        return 'neutral';
      });
    }

    return articles.map(() => 'neutral');
  } catch (error) {
    console.error(`  ❌ Batch error: ${error.message}`);
    stats.errors++;
    return articles.map(() => 'neutral');
  }
}

/**
 * Analyze sentiment of articles relative to a market question
 * Processes in small batches to keep prompt size manageable
 */
async function analyzeArticlesSentiment(articles, marketTitle) {
  if (articles.length === 0) return [];

  const allSentiments = [];

  // Process in batches of ARTICLES_PER_BATCH
  for (let i = 0; i < articles.length; i += ARTICLES_PER_BATCH) {
    const batch = articles.slice(i, i + ARTICLES_PER_BATCH);
    const batchSentiments = await analyzeSmallBatch(batch, marketTitle);
    allSentiments.push(...batchSentiments);
  }

  return allSentiments;
}

/**
 * Extract market title from marketId
 */
function marketIdToTitle(marketId) {
  return marketId
    .replace(/-\d+$/, '')
    .replace(/-/g, ' ')
    .replace(/\b\w/g, c => c.toUpperCase())
    + '?';
}

/**
 * Process a single news file
 */
async function processNewsFile(filename) {
  const filePath = path.join(NEWS_DIR, filename);
  const marketId = filename.replace('.json', '');

  try {
    const content = await fs.readFile(filePath, 'utf-8');
    const articles = JSON.parse(content);

    if (!Array.isArray(articles) || articles.length === 0) {
      return { marketId, articles: [], changed: false };
    }

    const marketTitle = marketIdToTitle(marketId);

    // Analyze sentiment
    const sentiments = await analyzeArticlesSentiment(articles, marketTitle);

    // Update articles
    let changed = false;
    articles.forEach((article, i) => {
      const newSentiment = sentiments[i] || 'neutral';
      if (article.sentiment !== newSentiment) {
        article.sentiment = newSentiment;
        stats.sentimentChanges[newSentiment]++;
        changed = true;
      }
    });

    stats.articlesProcessed += articles.length;

    // Save if changed and not dry run
    if (changed && !DRY_RUN) {
      await fs.writeFile(filePath, JSON.stringify(articles, null, 2));
    }

    return { marketId, articles, changed };
  } catch (error) {
    console.error(`  ❌ Error processing ${filename}: ${error.message}`);
    stats.errors++;
    return { marketId, articles: [], changed: false };
  }
}

/**
 * Update top-news.json with new sentiments
 */
async function updateTopNews(processedMarkets) {
  try {
    const content = await fs.readFile(TOP_NEWS_FILE, 'utf-8');
    const topNews = JSON.parse(content);

    // Create lookup map
    const articlesMap = new Map();
    for (const { articles } of processedMarkets) {
      for (const article of articles) {
        articlesMap.set(article.id, article.sentiment);
      }
    }

    // Update sentiments in top-news
    let updated = 0;
    for (const marketId of Object.keys(topNews.markets)) {
      for (const article of topNews.markets[marketId]) {
        const newSentiment = articlesMap.get(article.id);
        if (newSentiment && article.sentiment !== newSentiment) {
          article.sentiment = newSentiment;
          updated++;
        }
      }
    }

    topNews.updatedAt = new Date().toISOString();

    if (!DRY_RUN) {
      await fs.writeFile(TOP_NEWS_FILE, JSON.stringify(topNews, null, 2));
    }

    console.log(`  Updated ${updated} articles in top-news.json`);
  } catch (error) {
    console.error(`  ❌ Error updating top-news.json: ${error.message}`);
    stats.errors++;
  }
}

async function main() {
  console.log('🔧 Fix Sentiment Script');
  console.log('=======================');
  console.log(`Model: ${OPENAI_MODEL}`);
  console.log(`Max tokens: ${MAX_COMPLETION_TOKENS}`);
  console.log(`Mode: ${DRY_RUN ? 'DRY RUN (no changes saved)' : 'LIVE'}`);
  if (LIMIT) console.log(`Limit: ${LIMIT} markets`);
  console.log('');

  if (!process.env.OPENAI_API_KEY) {
    console.error('❌ Error: OPENAI_API_KEY environment variable is required');
    console.log('\nUsage: OPENAI_API_KEY=sk-xxx node fix-sentiment.mjs');
    process.exit(1);
  }

  // Get list of news files
  let files = (await fs.readdir(NEWS_DIR)).filter(f => f.endsWith('.json'));
  console.log(`📁 Found ${files.length} news files`);

  if (LIMIT) {
    files = files.slice(0, LIMIT);
    console.log(`📊 Processing limited to ${files.length} files`);
  }

  console.log('\n🚀 Starting sentiment analysis...\n');

  // Process files in batches
  const allResults = [];
  for (let i = 0; i < files.length; i += BATCH_SIZE) {
    const batch = files.slice(i, i + BATCH_SIZE);

    const results = await Promise.all(batch.map(f => processNewsFile(f)));
    allResults.push(...results);
    stats.filesProcessed += batch.length;

    const pct = Math.round((stats.filesProcessed / files.length) * 100);
    const changed = results.filter(r => r.changed).length;
    console.log(`  Processed ${stats.filesProcessed}/${files.length} (${pct}%) - ${changed} files changed in batch`);

    // No delay - high RPM available
  }

  // Update top-news.json
  console.log('\n📝 Updating top-news.json...');
  await updateTopNews(allResults);

  // Print summary
  console.log('\n📊 Summary');
  console.log('==========');
  console.log(`Files processed: ${stats.filesProcessed}`);
  console.log(`Articles processed: ${stats.articlesProcessed}`);
  console.log(`Sentiment changes:`);
  console.log(`  - Bullish: ${stats.sentimentChanges.bullish}`);
  console.log(`  - Bearish: ${stats.sentimentChanges.bearish}`);
  console.log(`  - Neutral: ${stats.sentimentChanges.neutral}`);
  console.log(`Errors: ${stats.errors}`);

  if (DRY_RUN) {
    console.log('\n⚠️  This was a dry run. No files were modified.');
    console.log('   Run without --dry-run to save changes.');
  } else {
    console.log('\n✅ Done! Changes have been saved.');
    console.log('\nNext steps:');
    console.log('  cd polynews-news');
    console.log('  git add -A');
    console.log('  git commit -m "fix: Re-analyze sentiment with correct token count"');
    console.log('  git push');
  }
}

main().catch((error) => {
  console.error('Fatal error:', error);
  process.exit(1);
});
