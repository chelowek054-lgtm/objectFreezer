# Как работают ноды `object-freezer`

Регистрация нод: [`__init__.py`](../__init__.py). Реализации: [`nodes/`](../nodes/). Общая логика: [`core/`](../core/).

Файл [`freezer.py`](../freezer.py) только реэкспортирует классы из `nodes/` (на случай импорта по старому пути).

| ID в ComfyUI | Класс |
|--------------|--------|
| **Blueprint Creator (.blueprint)** | [`BlueprintCreator`](../nodes/blueprint_creator.py) |
| **Blueprint Injector (patch model)** | [`BlueprintInjector`](../nodes/blueprint_injector.py) |
| **Blueprint Path (Output)** | [`BlueprintPathOutput`](../nodes/blueprint_outputs.py) |

Пакет не модифицирует исходники ComfyUI: текстовые токены blueprint подмешиваются через **`ModelPatcher.set_model_post_input_patch`** (Flux `post_input`); обязательный **`reference_latent`** подмешивается в **`diffusion_model`** через **`WrappersMP.DIFFUSION_MODEL`**.

---

## 1. Blueprint Creator

Строит файл **`.blueprint`** (safetensors): `z_hyb`, текстовые токены **`blueprint_text_tokens`** (сырой выход TE без проекций), **`keyword_embedding`**, обязательный **`reference_latent`** `[1,C,H,W]` (float16), берётся из первого кадра входного `IMAGE`.

### Входы (сводка)

| Параметр | Смысл |
|----------|--------|
| `images` | Батч `IMAGE`. |
| `vae`, `text_encoder` | Как в основном пайплайне генерации. |
| `ollama_url`, `vlm_model` | Ollama для VLM; JSON с `summary_text`, `details`, при `character` — `face_description`. |
| `object_id`, `object_class` | Папка под `output/blueprints`, префикт текста, влияет на поля VLM. |
| `output_dir` | Базовый каталог (пусто — см. [`paths.py`](../core/paths.py)). |
| `seed` | Сиды проекций `R_sem`, `R_face`. |
| (нет) | Референс всегда сохраняется как **`reference_latent`**. |

Ключи тензоров в файле: `z_vision`, `z_sem`, `z_geo`, `z_face`, `z_hyb`, `keyword_embedding`, `blueprint_text_tokens`, **`reference_latent`**.

Текстовые токены: [`encode_blueprint_text_tokens_for_diffusion`](../core/text_encoder.py) → float32 CPU **`[1, T, D]`** (например Klein **D=12288**).

Запись: [`save_blueprint_safetensors`](../core/blueprint_io.py), индекс **`{object_id}.index.json`**: [`update_index_json`](../core/blueprint_io.py).

---

## 2. Blueprint Injector

- **`blueprint_text_tokens`**: [`load_blueprint_text_tokens`](../core/blueprint_io.py) — ключ **`blueprint_text_tokens`**. После **`txt_norm`** / **`txt_in`** конкатенируются к `txt` сцены; последняя размерность должна совпадать с **`txt_in.in_features`** (например 12288 для Klein).
- **`reference_latent`**: [`load_reference_latent`](../core/blueprint_io.py) — ключ **`reference_latent`**. Если выбран `.index.json`, то **`reference_frame_index`** выбирает `entries[]` (какой `.blueprint` взять). Подмешивание в **`ref_latents`** Flux; режим **`ref_latents_method`**: `default` → во Flux **`offset`**, `index_timestep_zero` → **`index_timestep_zero`**.

| Параметр | Смысл |
|----------|--------|
| `blueprint_path` | `.index.json` из dropdown или путь к `.blueprint`. |
| `blueprint_scale` | Множитель для текстовых и референс-латентов; `0` отключает оба пути. |
| `inject_reference_latent` | Включить ли референс из файла. |
| `reference_frame_index` | Какой слой стека `[N,C,H,W]` использовать. |
| `ref_latents_method` | `default` / `index_timestep_zero`. |
| `bp_tokens_position`, `keyword_position`, `seed_W`, `debug` | См. исходник ноды. |

Дополнительные именованные аргументы из графа игнорируются (`**_kwargs`).

---

## 3. Blueprint Path (Output)

Вход `blueprint_path`, вывод в UI — см. [`blueprint_outputs.py`](../nodes/blueprint_outputs.py).

---

## Файлы по умолчанию

`ComfyUI/output/blueprints/<object_id>/` — `.blueprint` и `.index.json`.

---

## Совместимость

Формат `.blueprint` зафиксирован перечисленными ключами; старые файлы с другими именами тензоров нужно пересоздать в Creator.
