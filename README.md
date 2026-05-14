# Viking Engine: От 179 до 7.3 секунд  Полная История

**Как consumer RTX 4090 обогнал серверные решения в обучении Flux2 LoRA**

*Orakul Studio — Chernihiv, Ukraine 🇺🇦*

---

## Финальная таблица. Вся эволюция.

| Версия | Ключевое изменение | Rank 512 | Ускорение |
|--------|-------------------|----------|-----------|
| Baseline | Стандартный ai-toolkit | **179 s/it** | 1× |
| Viking v1 | Double-buffer async CUDA | **37 s/it** | 4.8× |
| Viking v2 | + bf16 weight forcing | **14 s/it** | 12.8× |
| Oracle-60 | + Hardware FP8 + CPU prequant | **8.7 s/it** | 20.6× |
| **ЛЕГЕНДА** | **+ Full 8-bit stack + AdamW 8-bit** | **7.3 s/it** | **24.5×** |

**Железо:** RTX 4090 · i9-13900K · 128 GB RAM · Чернигов, Украина 🇺🇦  
**Модель:** Flux2-dev (32 миллиарда параметров)  
**Параметров обучается:** 3,120,562,176 при rank 512

---

## Что это значит на практике

```
2000 шагов обучения rank 512:

Baseline:    179 × 2000 = ~99 часов    ← невозможно использовать
Viking v1:    37 × 2000 = ~20 часов    ← уже реально
Viking v2:    14 × 2000 =  ~7 часов    ← рабочий режим
Oracle-60:   8.7 × 2000 =  ~4.8 часа  ← профессионально
ЛЕГЕНДА:     7.3 × 2000 =  ~4 часа    ← серверная скорость
```

---

## Четыре прорыва — четыре инсайта

### Прорыв 1 — Double Buffer (Viking v1)

Стандартный offloading: GPU ждёт веса. Веса едут пока GPU ждёт.

```
БЫЛО:    [compute N] → [idle] → [transfer N+1] → [compute N+1]
СТАЛО:   [compute N]
         [transfer N+1] ← параллельно в отдельном CUDA stream
```

Transfer исчез из профайлера полностью. CUDA Events синхронизируют
два потока без блокировки. Ping-pong буферы не дают потокам мешать
друг другу.

**Результат: 179 → 37 сек. 4.8×**

---

### Прорыв 2 — bf16 forcing (Viking v2)

Одна строка в `BaseSDTrainProcess.py`:

```python
# todo switch everything to proper mixed precision like this
self.network.force_to(self.device_torch, dtype=torch.bfloat16)
```

LoRA матрицы: float32 → bfloat16. Размер вдвое меньше.
PCIe трафик вдвое меньше. Transfer скрывается ещё легче.

Этот `# todo` комментарий был в оригинальном коде.
Мы его прочли и реализовали.

**Результат: 37 → 14 сек. 2.6×**

---

### Прорыв 3 — Protocol Oracle-60 (Hardware FP8)

RTX 4090 (Ada Lovelace, sm_89) имеет нативные FP8 Tensor Cores.
До Oracle-60 они не использовались в ai-toolkit вообще.

```
>>> [ORACLE-60] HARDWARE FP8 ACTIVE

CPU pre-quantization: i9-13900K квантует блоки в FP8 ДО загрузки в GPU
PCIe трафик: BF16 (2 байт/параметр) → FP8 (1 байт/параметр)
```

CPU делает тяжёлую подготовку. GPU получает готовые данные.
PCIe шина разблокирована. Энергопотребление: ~300W вместо 450W.

Alpha 512 при Rank 512 (Scale 1.0) — режим который раньше вешал систему —
теперь работает стабильно.

**Результат: 14 → 8.7 сек. 1.6×**

---

### Прорыв 4 — ЛЕГЕНДА (Full 8-bit Stack)

Полный стек 8-bit квантования:

```
Model weights:       BF8   (qtype: bf8)
Text encoder:        BF8   (quantize_te: bf8)
Weight transfer:     FP8   (Oracle-60)
Optimizer moments:   INT8  (adamw8bit)
```

AdamW 8-bit считается «самым медленным» оптимизатором.
Но посмотрите на лог после шага 100:

```
Step 99:  35.83 s/it  ← optimizer калибрует INT8 диапазоны
Step 100: 35.57 s/it  ← калибровка завершена
Step 101: 35.29 s/it  ↓
Step 110: 33.01 s/it  ↓  каждый шаг быстрее
Step 130: 25.xx s/it  ↓
Step 180: 7.28 s/it   ← стабильно ✓
```

**Почему это происходит:**

AdamW хранит 2 вектора моментов (m и v) на каждый параметр.
В FP32 это: 3.12B × 4 байта × 2 = **~25 GB** — невозможно.
В INT8 это: 3.12B × 1 байт × 2 = **~6 GB** — влезает.

После калибровки (~100 шагов) 19 GB освобождается резко.
VRAM перестаёт задыхаться. Pipeline летит.

И при этом AdamW даёт **лучшее качество градиентов** чем Adafactor.
«Самый медленный» оказался самым быстрым и самым точным.

**Результат: 8.7 → 7.3 сек. 1.2×**

---

## Профиль идеальной итерации (ЛЕГЕНДА)

```
train_loop:    7.28s avg
backward:      4.52s avg  ← GPU градиенты в FP8 Tensor Cores
predict_unet:  2.47s avg  ← forward pass
optimizer_step: 0.29s avg ← AdamW 8-bit (INT8 моменты)
transfer:       0.00s     ← скрыт полностью ✓
```

---

## Почему это важно за пределами RTX 4090

```python
# Consumer (этот репо):
weight_cpu.to(device, non_blocking=True)

# Server NVLink:
weight_gpu0.to(device_1, non_blocking=True)

# Tensor parallelism:
weight_shard.to(target_device, non_blocking=True)
```

Та же двойная буферизация. Те же CUDA Streams. Те же Events.
Тот же полный 8-bit стек.

На H100/A100 кластерах эта архитектура устраняет inter-GPU
transfer stalls в tensor parallelism — точно так же как она
устраняет CPU-GPU transfer stalls на consumer железе.

---

## Доказательства

**[Логи:](https://github.com/OrakulStudio/-consumer-RTX-4090-Flux2-LoRA/tree/main/Logs)** все файлы в репозитории с именами ПОБЕДА / ЛЕГЕНДА  
**[Видео:](https://youtu.be/7zPQvcNFFnc)** реальный терминал, цифры в реальном времени  
**[Модели:](https://civitai.com/models/2619018/aivazovsky-smerti-net-genij-vnov-vzyalsya-za-kist-or-lora-rank-1024-flux2-dev-128-paintings-orakul-studio-publikuyutsya-dve-versii-400-shagov-i-500-shagov)** обученные с Viking Engine опубликованы на CivitAI

🎨 [AI_vazovsky на CivitAI](https://civitai.com/user/orakul_storm) —
Rank 1024, 128 картин Айвазовского. Включая авторскую подпись
которую модель ставит сама.

---

## Контекст

[ostris](https://github.com/ostris) — автор ai-toolkit используемого
тысячами людей — запросил этот код для интеграции. Тикет открыт.

Весь путь от 179 до 7.3 секунд пройден:
- На одной RTX 4090
- В Чернигове
- В условиях лимитов по электричеству
- Под звуки воздушной тревоги

**Правильная архитектура важнее железа.**  
**Необходимость — лучший инженер.**

---

## Код

🐙 [GitHub — OrakulStudio](https://github.com/OrakulStudio)  
🤗 [Hugging Face — OrakulStorm](https://huggingface.co/OrakulStorm)  
🎨 [CivitAI — orakul_storm](https://civitai.com/user/orakul_storm)

---

*Запах утюга стабільний. Система працює. 🦊⚡*

*Chernihiv, Ukraine 🇺🇦 · Orakul Studio · 2026*
