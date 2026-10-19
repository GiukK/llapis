#include "server-tools.h"

#include "http.h"   // common_http_client (web_fetch / web_search, fork addition M6d)

#include <sheredom/subprocess.h>

#include <filesystem>
#include <fstream>
#include <regex>
#include <thread>
#include <chrono>
#include <atomic>
#include <cstring>
#include <climits>
#include <algorithm>
#include <unordered_set>

namespace fs = std::filesystem;

//
// internal helpers
//

static std::vector<char *> to_cstr_vec(const std::vector<std::string> & v) {
    std::vector<char *> r;
    r.reserve(v.size() + 1);
    for (const auto & s : v) {
        r.push_back(const_cast<char *>(s.c_str()));
    }
    r.push_back(nullptr);
    return r;
}

struct run_proc_result {
    std::string output;
    int  exit_code = -1;
    bool timed_out = false;
};

static run_proc_result run_process(
        const std::vector<std::string> & args,
        size_t max_output,
        int timeout_secs) {
    run_proc_result res;

    subprocess_s proc;
    auto argv = to_cstr_vec(args);

    int options = subprocess_option_no_window
                | subprocess_option_combined_stdout_stderr
                | subprocess_option_inherit_environment
                | subprocess_option_search_user_path;

    if (subprocess_create(argv.data(), options, &proc) != 0) {
        res.output = "failed to spawn process";
        return res;
    }

    std::atomic<bool> done{false};
    std::atomic<bool> timed_out{false};

    std::thread timeout_thread([&]() {
        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_secs);
        while (!done.load()) {
            if (std::chrono::steady_clock::now() >= deadline) {
                timed_out.store(true);
                subprocess_terminate(&proc);
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    });

    FILE * f = subprocess_stdout(&proc);
    std::string output;
    bool truncated = false;
    if (f) {
        char buf[4096];
        while (fgets(buf, sizeof(buf), f) != nullptr) {
            if (!truncated) {
                size_t len = strlen(buf);
                if (output.size() + len <= max_output) {
                    output.append(buf, len);
                } else {
                    output.append(buf, max_output - output.size());
                    truncated = true;
                }
            }
        }
    }

    done.store(true);
    if (timeout_thread.joinable()) {
        timeout_thread.join();
    }

    subprocess_join(&proc, &res.exit_code);
    subprocess_destroy(&proc);

    res.output    = output;
    res.timed_out = timed_out.load();
    if (truncated) {
        res.output += "\n[output truncated]";
    }
    return res;
}

json server_tool::to_json() {
    return {
        {"display_name", display_name},
        {"tool", name},
        {"type", "builtin"},
        {"permissions", json{
            {"write", permission_write}
        }},
        {"definition", get_definition()},
    };
}

//
// read_file: read a file with optional line range and line-number prefix
//

static constexpr size_t SERVER_TOOL_READ_FILE_MAX_SIZE = 16 * 1024; // 16 KB

struct server_tool_read_file : server_tool {
    server_tool_read_file() {
        name = "read_file";
        display_name = "Read file";
        permission_write = false;
    }

    json get_definition() override {
        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description", "Read the contents of a file. Optionally specify a 1-based line range. "
                                "If append_loc is true, each line is prefixed with its line number (e.g. \"1\u2192 ...\")."},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"path",       {{"type", "string"},  {"description", "Path to the file"}}},
                        {"start_line", {{"type", "integer"}, {"description", "First line to read, 1-based (default: 1)"}}},
                        {"end_line",   {{"type", "integer"}, {"description", "Last line to read, 1-based inclusive (default: end of file)"}}},
                        {"append_loc", {{"type", "boolean"}, {"description", "Prefix each line with its line number"}}},
                    }},
                    {"required", json::array({"path"})},
                }},
            }},
        };
    }

    json invoke(json params) override {
        std::string path  = params.at("path").get<std::string>();
        int  start_line   = json_value(params, "start_line", 1);
        int  end_line     = json_value(params, "end_line",  -1); // -1 = no limit
        bool append_loc   = json_value(params, "append_loc", false);

        std::error_code ec;
        uintmax_t file_size = fs::file_size(path, ec);
        if (ec) {
            return {{"error", "cannot stat file: " + ec.message()}};
        }
        if (file_size > SERVER_TOOL_READ_FILE_MAX_SIZE && end_line == -1) {
            return {{"error", string_format(
                "file too large (%zu bytes, max %zu). Use start_line/end_line to read a portion.",
                (size_t)file_size, SERVER_TOOL_READ_FILE_MAX_SIZE)}};
        }

        std::ifstream f(path);
        if (!f) {
            return {{"error", "failed to open file: " + path}};
        }

        std::string result;
        std::string line;
        int lineno = 0;

        while (std::getline(f, line)) {
            lineno++;
            if (lineno < start_line) continue;
            if (end_line != -1 && lineno > end_line) break;

            std::string out_line;
            if (append_loc) {
                out_line = std::to_string(lineno) + "\u2192 " + line + "\n";
            } else {
                out_line = line + "\n";
            }

            if (result.size() + out_line.size() > SERVER_TOOL_READ_FILE_MAX_SIZE) {
                result += "[output truncated]";
                break;
            }
            result += out_line;
        }

        return {{"plain_text_response", result}};
    }
};

//
// file_glob_search: find files matching a glob pattern under a base directory
//

static constexpr size_t SERVER_TOOL_FILE_SEARCH_MAX_RESULTS = 100;

struct server_tool_file_glob_search : server_tool {
    server_tool_file_glob_search() {
        name = "file_glob_search";
        display_name = "File search";
        permission_write = false;
    }

    json get_definition() override {
        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description", "Recursively search for files matching a glob pattern under a directory."},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"path",    {{"type", "string"}, {"description", "Base directory to search in"}}},
                        {"include", {{"type", "string"}, {"description", "Glob pattern for files to include (e.g. \"**/*.cpp\"). Default: **"}}},
                        {"exclude", {{"type", "string"}, {"description", "Glob pattern for files to exclude"}}},
                    }},
                    {"required", json::array({"path"})},
                }},
            }},
        };
    }

    json invoke(json params) override {
        std::string base    = params.at("path").get<std::string>();
        std::string include = json_value(params, "include", std::string("**"));
        std::string exclude = json_value(params, "exclude", std::string(""));

        std::ostringstream output_text;
        size_t count = 0;

        std::error_code ec;
        for (const auto & entry : fs::recursive_directory_iterator(base,
                fs::directory_options::skip_permission_denied, ec)) {
            if (!entry.is_regular_file()) continue;

            std::string rel = fs::relative(entry.path(), base, ec).string();
            if (ec) continue;
            std::replace(rel.begin(), rel.end(), '\\', '/');

            if (!glob_match(include, rel)) continue;
            if (!exclude.empty() && glob_match(exclude, rel)) continue;

            output_text << entry.path().string() << "\n";
            if (++count >= SERVER_TOOL_FILE_SEARCH_MAX_RESULTS) {
                break;
            }
        }

        output_text << "\n---\nTotal matches: " << count << "\n";

        return {{"plain_text_response", output_text.str()}};
    }
};

//
// grep_search: search for a regex pattern in files
//

static constexpr size_t SERVER_TOOL_GREP_SEARCH_MAX_RESULTS = 100;

struct server_tool_grep_search : server_tool {
    server_tool_grep_search() {
        name = "grep_search";
        display_name = "Grep search";
        permission_write = false;
    }

    json get_definition() override {
        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description", "Search for a regex pattern in files under a path. Returns matching lines."},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"path",                {{"type", "string"},  {"description", "File or directory to search in"}}},
                        {"pattern",             {{"type", "string"},  {"description", "Regular expression pattern to search for"}}},
                        {"include",             {{"type", "string"},  {"description", "Glob pattern to filter files (default: **)"}}},
                        {"exclude",             {{"type", "string"},  {"description", "Glob pattern to exclude files"}}},
                        {"return_line_numbers", {{"type", "boolean"}, {"description", "If true, include line numbers in results"}}},
                    }},
                    {"required", json::array({"path", "pattern"})},
                }},
            }},
        };
    }

    json invoke(json params) override {
        std::string path    = params.at("path").get<std::string>();
        std::string pat_str = params.at("pattern").get<std::string>();
        std::string include = json_value(params, "include", std::string("**"));
        std::string exclude = json_value(params, "exclude", std::string(""));
        bool show_lineno    = json_value(params, "return_line_numbers", false);

        std::regex pattern;
        try {
            pattern = std::regex(pat_str);
        } catch (const std::regex_error & e) {
            return {{"error", std::string("invalid regex: ") + e.what()}};
        }

        std::ostringstream output_text;
        size_t total = 0;

        auto search_file = [&](const fs::path & fpath) {
            std::ifstream f(fpath);
            if (!f) return;
            std::string line;
            int lineno = 0;
            while (std::getline(f, line) && total < SERVER_TOOL_GREP_SEARCH_MAX_RESULTS) {
                lineno++;
                if (std::regex_search(line, pattern)) {
                    output_text << fpath.string() << ":";
                    if (show_lineno) {
                        output_text << lineno << ":";
                    }
                    output_text << line << "\n";
                    total++;
                }
            }
        };

        std::error_code ec;
        if (fs::is_regular_file(path, ec)) {
            search_file(path);
        } else if (fs::is_directory(path, ec)) {
            for (const auto & entry : fs::recursive_directory_iterator(path,
                    fs::directory_options::skip_permission_denied, ec)) {
                if (!entry.is_regular_file()) continue;
                if (total >= SERVER_TOOL_GREP_SEARCH_MAX_RESULTS) break;

                std::string rel = fs::relative(entry.path(), path, ec).string();
                if (ec) continue;
                std::replace(rel.begin(), rel.end(), '\\', '/');

                if (!glob_match(include, rel)) continue;
                if (!exclude.empty() && glob_match(exclude, rel)) continue;

                search_file(entry.path());
            }
        } else {
            return {{"error", "path does not exist: " + path}};
        }

        output_text << "\n\n---\nTotal matches: " << total << "\n";

        return {{"plain_text_response", output_text.str()}};
    }
};

//
// exec_shell_command: run an arbitrary shell command
//

static constexpr size_t SERVER_TOOL_EXEC_SHELL_COMMAND_MAX_OUTPUT_SIZE = 16 * 1024; // 16 KB
static constexpr int    SERVER_TOOL_EXEC_SHELL_COMMAND_MAX_TIMEOUT     = 60;        // seconds

struct server_tool_exec_shell_command : server_tool {
    server_tool_exec_shell_command() {
        name = "exec_shell_command";
        display_name = "Execute shell command";
        permission_write = true;
    }

    json get_definition() override {
        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description", "Execute a shell command and return its output (stdout and stderr combined)."},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"command",         {{"type", "string"},  {"description", "Shell command to execute"}}},
                        {"timeout",         {{"type", "integer"}, {"description", string_format("Timeout in seconds (default 10, max %d)", SERVER_TOOL_EXEC_SHELL_COMMAND_MAX_TIMEOUT)}}},
                        {"max_output_size", {{"type", "integer"}, {"description", string_format("Maximum output size in bytes (default %zu)", SERVER_TOOL_EXEC_SHELL_COMMAND_MAX_OUTPUT_SIZE)}}},
                    }},
                    {"required", json::array({"command"})},
                }},
            }},
        };
    }

    json invoke(json params) override {
        std::string command   = params.at("command").get<std::string>();
        int    timeout        = json_value(params, "timeout",         10);
        size_t max_output     = (size_t) json_value(params, "max_output_size", (int) SERVER_TOOL_EXEC_SHELL_COMMAND_MAX_OUTPUT_SIZE);

        timeout    = std::min(timeout,    SERVER_TOOL_EXEC_SHELL_COMMAND_MAX_TIMEOUT);
        max_output = std::min(max_output, SERVER_TOOL_EXEC_SHELL_COMMAND_MAX_OUTPUT_SIZE);

#ifdef _WIN32
        std::vector<std::string> args = {"cmd", "/c", command};
#else
        std::vector<std::string> args = {"sh", "-c", command};
#endif

        auto res = run_process(args, max_output, timeout);

        std::string text_output = res.output;
        text_output += string_format("\n[exit code: %d]", res.exit_code);
        if (res.timed_out) {
            text_output += " [exit due to timed out]";
        }

        return {{"plain_text_response", text_output}};
    }
};

//
// write_file: create or overwrite a file
//

struct server_tool_write_file : server_tool {
    server_tool_write_file() {
        name = "write_file";
        display_name = "Write file";
        permission_write = true;
    }

    json get_definition() override {
        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description", "Write content to a file, creating it (including parent directories) if it does not exist. May use with edit_file for more complex edits."},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"path",    {{"type", "string"}, {"description", "Path of the file to write"}}},
                        {"content", {{"type", "string"}, {"description", "Content to write"}}},
                    }},
                    {"required", json::array({"path", "content"})},
                }},
            }},
        };
    }

    json invoke(json params) override {
        std::string path    = params.at("path").get<std::string>();
        std::string content = params.at("content").get<std::string>();

        std::error_code ec;
        fs::path fpath(path);
        if (fpath.has_parent_path()) {
            fs::create_directories(fpath.parent_path(), ec);
            if (ec) {
                return {{"error", "failed to create directories: " + ec.message()}};
            }
        }

        std::ofstream f(path, std::ios::binary);
        if (!f) {
            return {{"error", "failed to open file for writing: " + path}};
        }
        f << content;
        if (!f) {
            return {{"error", "failed to write file: " + path}};
        }

        return {{"result", "file written successfully"}, {"path", path}, {"bytes", content.size()}};
    }
};

//
// edit_file: edit file content via line-based changes
//

struct server_tool_edit_file : server_tool {
    server_tool_edit_file() {
        name = "edit_file";
        display_name = "Edit file";
        permission_write = true;
    }

    json get_definition() override {
        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description",
                    "Edit a file by applying a list of line-based changes. "
                    "Each change targets a 1-based inclusive line range and has a mode: "
                    "\"replace\" (replace lines with content), "
                    "\"delete\" (remove lines, content must be empty string), "
                    "\"append\" (insert content after line_end). "
                    "Set line_start to -1 to target the end of file (line_end is ignored in that case). "
                    "Changes must not overlap. They are applied in reverse line order automatically."},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"path",    {{"type", "string"}, {"description", "Path to the file to edit"}}},
                        {"changes", {
                            {"type", "array"},
                            {"description", "List of changes to apply"},
                            {"items", {
                                {"type", "object"},
                                {"properties", {
                                    {"mode",       {{"type", "string"},  {"description", "\"replace\", \"delete\", or \"append\""}}},
                                    {"line_start", {{"type", "integer"}, {"description", "First line of the range (1-based); use -1 for end of file"}}},
                                    {"line_end",   {{"type", "integer"}, {"description", "Last line of the range (1-based, inclusive); ignored when line_start is -1"}}},
                                    {"content",    {{"type", "string"},  {"description", "Content to insert; must be empty string for delete mode"}}},
                                }},
                                {"required", json::array({"mode", "line_start", "line_end", "content"})},
                            }},
                        }},
                    }},
                    {"required", json::array({"path", "changes"})},
                }},
            }},
        };
    }

    json invoke(json params) override {
        std::string path = params.at("path").get<std::string>();
        const json & changes = params.at("changes");

        if (!changes.is_array()) {
            return {{"error", "\"changes\" must be an array"}};
        }

        // read file into lines
        std::ifstream fin(path);
        if (!fin) {
            return {{"error", "failed to open file: " + path}};
        }
        std::vector<std::string> lines;
        {
            std::string line;
            while (std::getline(fin, line)) {
                lines.push_back(line);
            }
        }
        fin.close();

        // validate and collect changes, then sort descending by line_start
        struct change_entry {
            std::string mode;
            int line_start; // 1-based
            int line_end;   // 1-based inclusive
            std::string content;
        };
        std::vector<change_entry> entries;
        entries.reserve(changes.size());

        for (const auto & ch : changes) {
            change_entry e;
            e.mode       = ch.at("mode").get<std::string>();
            e.line_start = ch.at("line_start").get<int>();
            e.line_end   = ch.at("line_end").get<int>();
            e.content    = ch.at("content").get<std::string>();

            if (e.mode != "replace" && e.mode != "delete" && e.mode != "append") {
                return {{"error", "invalid mode \"" + e.mode + "\"; must be replace, delete, or append"}};
            }
            if (e.mode == "delete" && !e.content.empty()) {
                return {{"error", "content must be empty string for delete mode"}};
            }
            int n = (int) lines.size();
            if (e.line_start == -1) {
                // -1 targets end of file -> valid for append only; line_end is ignored
                if (e.mode != "append") {
                    return {{"error", "line_start -1 (end of file) is only valid for append mode"}};
                }
                // append at end of file: insert position is the current line count
                e.line_start = n;
                e.line_end   = n;
            } else {
                if (e.line_start < 1 || e.line_end < e.line_start) {
                    return {{"error", string_format("invalid line range [%d, %d]", e.line_start, e.line_end)}};
                }
                if (e.line_end > n) {
                    return {{"error", string_format("line_end %d exceeds file length %d", e.line_end, n)}};
                }
            }
            entries.push_back(std::move(e));
        }

        // sort descending so earlier-indexed changes don't shift later ones
        std::sort(entries.begin(), entries.end(), [](const change_entry & a, const change_entry & b) {
            return a.line_start > b.line_start;
        });

        // apply changes (0-based indices internally)
        for (const auto & e : entries) {
            int idx_start = e.line_start - 1; // 0-based
            int idx_end   = e.line_end   - 1; // 0-based inclusive

            // split content into lines (preserve trailing newline awareness)
            std::vector<std::string> new_lines;
            if (!e.content.empty()) {
                std::istringstream ss(e.content);
                std::string ln;
                while (std::getline(ss, ln)) {
                    new_lines.push_back(ln);
                }
                // if content ends with \n, getline consumed it — no extra empty line needed
                // if content does NOT end with \n, last line is still captured correctly
            }

            if (e.mode == "replace") {
                // erase [idx_start, idx_end] and insert new_lines
                lines.erase(lines.begin() + idx_start, lines.begin() + idx_end + 1);
                lines.insert(lines.begin() + idx_start, new_lines.begin(), new_lines.end());
            } else if (e.mode == "delete") {
                lines.erase(lines.begin() + idx_start, lines.begin() + idx_end + 1);
            } else { // append
                // insert after idx_end; idx_end + 1 == lines.size() for end-of-file append
                lines.insert(lines.begin() + (idx_end + 1), new_lines.begin(), new_lines.end());
            }
        }

        // write file back
        std::ofstream fout(path, std::ios::binary);
        if (!fout) {
            return {{"error", "failed to open file for writing: " + path}};
        }
        for (size_t i = 0; i < lines.size(); i++) {
            fout << lines[i];
            if (i + 1 < lines.size()) {
                fout << "\n";
            }
        }
        if (!lines.empty()) {
            fout << "\n";
        }
        if (!fout) {
            return {{"error", "failed to write file: " + path}};
        }

        return {{"result", "file edited successfully"}, {"path", path}, {"lines", (int) lines.size()}};
    }
};

//
// apply_diff: apply a unified diff via git apply
//

struct server_tool_apply_diff : server_tool {
    server_tool_apply_diff() {
        name = "apply_diff";
        display_name = "Apply diff";
        permission_write = true;
    }

    json get_definition() override {
        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description", "Apply a unified diff to edit one or more files using git apply. Use this instead of edit_file when the changes are complex."},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"diff", {{"type", "string"}, {"description", "Unified diff content in git diff format"}}},
                    }},
                    {"required", json::array({"diff"})},
                }},
            }},
        };
    }

    json invoke(json params) override {
        std::string diff = params.at("diff").get<std::string>();

        // write diff to a temporary file
        static std::atomic<int> counter{0};
        std::string tmp_path = (fs::temp_directory_path() /
            ("llama_patch_" + std::to_string(++counter) + ".patch")).string();

        {
            std::ofstream f(tmp_path, std::ios::binary);
            if (!f) {
                return {{"error", "failed to create temp patch file"}};
            }
            f << diff;
        }

        auto res = run_process({"git", "apply", tmp_path}, 4096, 10);

        std::error_code ec;
        fs::remove(tmp_path, ec);

        if (res.exit_code != 0) {
            return {{"error", "git apply failed (exit " + std::to_string(res.exit_code) + "): " + res.output}};
        }
        return {{"result", "patch applied successfully"}};
    }
};

//
// get_datetime: returns the current date and time
//

struct server_tool_get_datetime : server_tool {
    server_tool_get_datetime() {
        name = "get_datetime";
        display_name = "Get Date & Time";
        permission_write = false;
    }

    json get_definition() override {
        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description", "Returns the current date and time"},
            }},
        };
    }

    json invoke(json) override {
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);

        return {{"result", std::ctime(&time)}};
    }
};

//
// web_fetch / web_search (LLapis fork addition).
// Reuse discipline: transport = common_http_client (common/http.h, the same
// httplib client the model downloader uses; HTTPS needs an SSL build, e.g.
// -DLLAMA_BUILD_BORINGSSL=ON). No new dependencies — HTML is reduced to text
// with plain LINEAR string scans (std::regex is banned on web content: these
// bytes are attacker-controlled and backtracking regexes are a CPU DoS).
// SSRF hardening (2026-07-08): model-chosen URLs must resolve to PUBLIC
// addresses at every redirect hop (web_check_host_public), downloads are
// size-capped in streaming (SERVER_TOOL_WEB_MAX_DOWNLOAD).
//

static constexpr size_t SERVER_TOOL_WEB_FETCH_DEFAULT_CHARS = 8 * 1024;
static constexpr size_t SERVER_TOOL_WEB_FETCH_MAX_CHARS     = 32 * 1024;

// decode the handful of HTML entities that actually matter for readability
static std::string web_decode_entities(std::string s) {
    static const std::pair<const char *, const char *> ents[] = {
        {"&nbsp;", " "}, {"&amp;", "&"}, {"&lt;", "<"}, {"&gt;", ">"},
        {"&quot;", "\""}, {"&#39;", "'"}, {"&#x27;", "'"}, {"&rsquo;", "'"},
        {"&lsquo;", "'"}, {"&rdquo;", "\""}, {"&ldquo;", "\""}, {"&mdash;", "-"},
        {"&ndash;", "-"}, {"&hellip;", "..."},
    };
    for (const auto & [from, to] : ents) {
        size_t pos = 0;
        const size_t flen = strlen(from);
        while ((pos = s.find(from, pos)) != std::string::npos) {
            s.replace(pos, flen, to);
            pos += strlen(to);
        }
    }
    return s;
}

// crude but dependency-free HTML -> text: drop script/style/head blocks and
// comments, turn block-closing tags into newlines, strip every other tag,
// decode entities, collapse whitespace runs.
// SECURITY: single linear scan, NO std::regex — backtracking regexes over
// attacker-controlled HTML are a CPU DoS (catastrophic backtracking).
static std::string web_html_to_text(const std::string & html) {
    auto lower_eq = [](const std::string & s, size_t pos, const char * lit) {
        for (size_t i = 0; lit[i]; ++i) {
            if (pos + i >= s.size() || std::tolower((unsigned char) s[pos + i]) != lit[i]) return false;
        }
        return true;
    };
    // case-insensitive find of `lit` (must be lowercase) starting at pos; npos if absent
    auto ifind = [&](const std::string & s, size_t pos, const char * lit) {
        const size_t n = strlen(lit);
        if (s.size() < n) return std::string::npos;
        for (size_t i = pos; i + n <= s.size(); ++i) {
            if (lower_eq(s, i, lit)) return i;
        }
        return std::string::npos;
    };
    std::string s;
    s.reserve(html.size() / 2);
    size_t i = 0;
    while (i < html.size()) {
        const char c = html[i];
        if (c != '<') { s += c; ++i; continue; }
        // comment
        if (lower_eq(html, i, "<!--")) {
            const size_t end = html.find("-->", i + 4);
            if (end == std::string::npos) break;
            i = end + 3; s += ' '; continue;
        }
        // container whose CONTENT must be dropped, not just the tags
        const char * skip_tag = nullptr;
        const char * skip_end = nullptr;
        if      (lower_eq(html, i, "<script")) { skip_tag = "<script"; skip_end = "</script"; }
        else if (lower_eq(html, i, "<style"))  { skip_tag = "<style";  skip_end = "</style";  }
        else if (lower_eq(html, i, "<head"))   { skip_tag = "<head";   skip_end = "</head";   }
        if (skip_tag) {
            // "<head" must not match "<header>": next char has to close the tag name
            const size_t after = i + strlen(skip_tag);
            const char nc = after < html.size() ? html[after] : '>';
            if (nc == '>' || nc == ' ' || nc == '\t' || nc == '\n' || nc == '\r' || nc == '/') {
                const size_t end = ifind(html, after, skip_end);
                if (end == std::string::npos) break;
                const size_t gt = html.find('>', end);
                if (gt == std::string::npos) break;
                i = gt + 1; s += ' '; continue;
            }
        }
        // block-closing tags and <br> become newlines; every other tag a space
        static const char * breaks[] = {
            "</p", "</div", "</li", "</tr", "</h1", "</h2", "</h3", "</h4", "</h5", "</h6",
            "</blockquote", "</pre", "</table", "<br",
        };
        bool is_break = false;
        for (const char * b : breaks) {
            if (lower_eq(html, i, b)) {
                const size_t after = i + strlen(b);
                const char nc = after < html.size() ? html[after] : '>';
                if (nc == '>' || nc == ' ' || nc == '/' || nc == '\t' || nc == '\n') { is_break = true; break; }
            }
        }
        const size_t gt = html.find('>', i);
        if (gt == std::string::npos) break;
        s += is_break ? '\n' : ' ';
        i = gt + 1;
    }
    s = web_decode_entities(s);
    // collapse: spaces/tabs runs -> one space; 3+ newlines -> two; trim line edges
    std::string out;
    out.reserve(s.size());
    int nl = 0, sp = 0;
    for (char c : s) {
        if (c == '\r') continue;
        if (c == '\n') { nl++; sp = 0; continue; }
        if (c == ' ' || c == '\t') { sp++; continue; }
        if (nl > 0) { out.append(nl >= 2 ? 2 : 1, '\n'); nl = 0; sp = 0; }
        else if (sp > 0 && !out.empty() && out.back() != '\n') { out += ' '; }
        sp = 0;
        out += c;
    }
    return out;
}

static std::string web_url_decode(const std::string & in) {
    std::string out;
    out.reserve(in.size());
    for (size_t i = 0; i < in.size(); ++i) {
        if (in[i] == '%' && i + 2 < in.size() && isxdigit((unsigned char) in[i+1]) && isxdigit((unsigned char) in[i+2])) {
            out += (char) std::stoi(in.substr(i + 1, 2), nullptr, 16);
            i += 2;
        } else if (in[i] == '+') {
            out += ' ';
        } else {
            out += in[i];
        }
    }
    return out;
}

static std::string web_url_encode(const std::string & in) {
    static const char hex[] = "0123456789ABCDEF";
    std::string out;
    for (unsigned char c : in) {
        if (isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') out += (char) c;
        else { out += '%'; out += hex[c >> 4]; out += hex[c & 15]; }
    }
    return out;
}

// SECURITY (SSRF): the model chooses these URLs. A fetch that can reach loopback,
// RFC1918/link-local ranges or cloud metadata endpoints turns the web tools into a
// bridge to every unauthenticated local service (including this server's own /tools).
// Every hop (initial URL + each redirect) must resolve to PUBLIC addresses only.
// Exemptions: the host of MOERT_WEB_SEARCH_URL (the user configured it deliberately,
// e.g. a SearXNG on the LAN), or MOERT_WEB_ALLOW_PRIVATE=1 to disable the check.
static bool web_sockaddr_is_private(const struct sockaddr * sa) {
    if (sa->sa_family == AF_INET) {
        const auto * a = (const struct sockaddr_in *) sa;
        const uint32_t ip = ntohl(a->sin_addr.s_addr);
        const uint8_t b0 = (uint8_t) (ip >> 24), b1 = (uint8_t) (ip >> 16);
        return b0 == 0 || b0 == 10 || b0 == 127 ||                    // 0/8, 10/8, loopback
               (b0 == 100 && b1 >= 64 && b1 <= 127) ||                // 100.64/10 (CGNAT)
               (b0 == 169 && b1 == 254) ||                            // 169.254/16 (link-local, cloud metadata)
               (b0 == 172 && b1 >= 16 && b1 <= 31) ||                 // 172.16/12
               (b0 == 192 && b1 == 168);                              // 192.168/16
    }
    if (sa->sa_family == AF_INET6) {
        const auto * a6 = (const struct sockaddr_in6 *) sa;
        const uint8_t * b = (const uint8_t *) &a6->sin6_addr;
        static const uint8_t loop6[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1};
        if (memcmp(b, loop6, 16) == 0) return true;                   // ::1
        if (memcmp(b, loop6, 15) == 0 && b[15] == 0) return true;     // ::
        if ((b[0] & 0xfe) == 0xfc) return true;                       // fc00::/7 (ULA)
        if (b[0] == 0xfe && (b[1] & 0xc0) == 0x80) return true;       // fe80::/10 (link-local)
        static const uint8_t v4map[12] = {0,0,0,0,0,0,0,0,0,0,0xff,0xff};
        if (memcmp(b, v4map, 12) == 0) {                              // ::ffff:a.b.c.d
            struct sockaddr_in v4{};
            v4.sin_family = AF_INET;
            memcpy(&v4.sin_addr, b + 12, 4);
            return web_sockaddr_is_private((const struct sockaddr *) &v4);
        }
        return false;
    }
    return true;   // unknown family: refuse
}

static void web_check_host_public(const common_http_url & parts) {
    if (const char * allow = std::getenv("MOERT_WEB_ALLOW_PRIVATE"); allow && allow[0] && allow[0] != '0') {
        return;
    }
    if (const char * tmpl = std::getenv("MOERT_WEB_SEARCH_URL")) {
        try {
            if (common_http_parse_url(tmpl).host == parts.host) return;   // user-configured engine
        } catch (...) {}
    }
    struct addrinfo hints{};
    hints.ai_family   = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    struct addrinfo * res = nullptr;
    if (getaddrinfo(parts.host.c_str(), nullptr, &hints, &res) != 0 || !res) {
        throw std::runtime_error("cannot resolve host: " + parts.host);
    }
    bool priv = false;
    for (const struct addrinfo * p = res; p; p = p->ai_next) {
        if (web_sockaddr_is_private(p->ai_addr)) { priv = true; break; }
    }
    freeaddrinfo(res);
    if (priv) {
        throw std::runtime_error("refused: '" + parts.host + "' resolves to a private/loopback address "
                                 "(local network fetch is blocked; set MOERT_WEB_ALLOW_PRIVATE=1 to override)");
    }
}

// hard cap on BYTES DOWNLOADED (not just on the text returned): without it a
// model-chosen URL to a multi-GB file buffers unbounded into RAM.
static constexpr size_t SERVER_TOOL_WEB_MAX_DOWNLOAD = 2 * 1024 * 1024;

// one GET with a browser-ish UA; returns body or throws. Redirects are followed
// MANUALLY (max 4 hops) so every hop re-passes the public-address check — httplib's
// automatic follow_location would happily land on 127.0.0.1 after hop 1.
static std::string web_http_get(std::string url, std::string * final_status = nullptr) {
    for (int hop = 0; hop < 4; ++hop) {
        const common_http_url pre = common_http_parse_url(url);   // validates scheme http/https
        web_check_host_public(pre);
        auto [cli, parts] = common_http_client(url);
        cli.set_follow_location(false);
        cli.set_connection_timeout(10, 0);
        cli.set_read_timeout(20, 0);
        cli.set_default_headers({
            {"User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) moert-server/1.0"},
            {"Accept", "text/html,application/xhtml+xml,application/json;q=0.9,text/plain;q=0.8,*/*;q=0.5"},
            {"Accept-Language", "en,it;q=0.8"},
        });
        std::string body;
        std::string location;
        int  status   = 0;
        bool over_cap = false;
        auto r = cli.Get(parts.path,
            [&](const httplib::Response & resp) {   // headers arrive before the body
                status   = resp.status;
                location = resp.get_header_value("Location");
                return true;
            },
            [&](const char * data, size_t n) {
                body.append(data, n);
                if (body.size() > SERVER_TOOL_WEB_MAX_DOWNLOAD) { over_cap = true; return false; }
                return true;
            });
        if (!r && !over_cap) {
            throw std::runtime_error("request failed: " + httplib::to_string(r.error()));
        }
        if (status >= 300 && status < 400 && !location.empty()) {
            std::string loc = location;
            if (loc.rfind("http://", 0) == 0 || loc.rfind("https://", 0) == 0) {
                url = loc;
            } else if (!loc.empty() && loc[0] == '/') {
                const bool tls = pre.scheme == "https";
                url = pre.scheme + "://" + pre.host +
                      ((tls && pre.port != 443) || (!tls && pre.port != 80) ? ":" + std::to_string(pre.port) : "") + loc;
            } else {
                throw std::runtime_error("unsupported redirect Location: " + loc);
            }
            continue;
        }
        if (final_status) *final_status = std::to_string(status);
        if (status < 200 || status >= 300) {
            throw std::runtime_error("HTTP " + std::to_string(status) + " for " + url);
        }
        return body;
    }
    throw std::runtime_error("too many redirects for " + url);
}

struct server_tool_web_fetch : server_tool {
    server_tool_web_fetch() {
        name = "web_fetch";
        display_name = "Fetch web page";
        permission_write = false;
    }

    json get_definition() override {
        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description", "Fetch a web page (http/https) and return its readable text content. "
                                "HTML is reduced to plain text; long pages are truncated."},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"url",       {{"type", "string"},  {"description", "Full URL, e.g. https://example.com/page"}}},
                        {"max_chars", {{"type", "integer"}, {"description", "Truncate the text to this many characters (default 8192, max 32768)"}}},
                    }},
                    {"required", json::array({"url"})},
                }},
            }},
        };
    }

    json invoke(json params) override {
        const std::string url = params.at("url").get<std::string>();
        size_t max_chars = (size_t) std::max(256, json_value(params, "max_chars", (int) SERVER_TOOL_WEB_FETCH_DEFAULT_CHARS));
        max_chars = std::min(max_chars, SERVER_TOOL_WEB_FETCH_MAX_CHARS);
        try {
            std::string status;
            std::string body = web_http_get(url, &status);
            // HTML -> text only when it looks like markup; JSON/plain pass through
            std::string text = body.find('<') != std::string::npos && body.find('>') != std::string::npos
                ? web_html_to_text(body) : body;
            bool truncated = text.size() > max_chars;
            if (truncated) text.resize(max_chars);
            std::string out = "URL: " + url + " (HTTP " + status + ")\n\n" + text;
            if (truncated) out += "\n[content truncated at " + std::to_string(max_chars) + " chars]";
            return {{"plain_text_response", out}};
        } catch (const std::exception & e) {
            return {{"error", std::string("web_fetch failed: ") + e.what()}};
        }
    }
};

struct server_tool_web_search : server_tool {
    server_tool_web_search() {
        name = "web_search";
        display_name = "Web search";
        permission_write = false;
    }

    json get_definition() override {
        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description", "Search the web and return a list of results (title, URL, snippet). "
                                "Use web_fetch on a result URL to read the page."},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"query",       {{"type", "string"},  {"description", "The search query"}}},
                        {"max_results", {{"type", "integer"}, {"description", "Number of results to return (default 5, max 10)"}}},
                    }},
                    {"required", json::array({"query"})},
                }},
            }},
        };
    }

    json invoke(json params) override {
        const std::string query = params.at("query").get<std::string>();
        const int max_results = std::min(10, std::max(1, json_value(params, "max_results", 5)));
        try {
            // MOERT_WEB_SEARCH_URL overrides the engine ({q} = url-encoded query); the
            // page is returned as stripped text (SearXNG & co.). Default: DuckDuckGo's
            // HTML endpoint, parsed into structured results.
            if (const char * tmpl = std::getenv("MOERT_WEB_SEARCH_URL")) {
                std::string url = tmpl;
                const size_t at = url.find("{q}");
                if (at == std::string::npos) throw std::runtime_error("MOERT_WEB_SEARCH_URL has no {q} placeholder");
                url.replace(at, 3, web_url_encode(query));
                std::string text = web_html_to_text(web_http_get(url));
                if (text.size() > SERVER_TOOL_WEB_FETCH_DEFAULT_CHARS) text.resize(SERVER_TOOL_WEB_FETCH_DEFAULT_CHARS);
                return {{"plain_text_response", "Search results (custom engine) for: " + query + "\n\n" + text}};
            }

            const std::string html = web_http_get("https://html.duckduckgo.com/html/?q=" + web_url_encode(query));
            // results: <a ... class="result__a" href="//duckduckgo.com/l/?uddg=<enc>&rut=...">Title</a>
            //          <a ... class="result__snippet" ...>snippet…</a>
            // SECURITY: parsed with plain string scans, NO std::regex — backtracking
            // patterns over an attacker-influenceable page are a CPU DoS.
            // scan_anchors: find each <a ...class="<marker>"...>, return (href, inner text)
            auto scan_anchors = [](const std::string & doc, const char * marker,
                                   std::vector<std::string> & hrefs_out, std::vector<std::string> & texts_out) {
                size_t pos = 0;
                while ((pos = doc.find(marker, pos)) != std::string::npos) {
                    const size_t mark = pos;
                    pos += strlen(marker);
                    // the enclosing <a ...> open tag: last "<a" before the marker, closed after it
                    const size_t open = doc.rfind("<a", mark);
                    if (open == std::string::npos) continue;
                    const size_t tag_end = doc.find('>', mark);
                    if (tag_end == std::string::npos) break;
                    const std::string tag = doc.substr(open, tag_end - open + 1);
                    std::string href;
                    const size_t h = tag.find("href=\"");
                    if (h != std::string::npos) {
                        const size_t hq = tag.find('"', h + 6);
                        if (hq != std::string::npos) href = tag.substr(h + 6, hq - (h + 6));
                    }
                    const size_t close = doc.find("</a>", tag_end);
                    if (close == std::string::npos) break;
                    hrefs_out.push_back(std::move(href));
                    texts_out.push_back(doc.substr(tag_end + 1, close - tag_end - 1));
                    pos = close + 4;
                }
            };
            std::vector<std::string> titles, urls, snippets;
            {
                std::vector<std::string> hrefs, texts;
                scan_anchors(html, "class=\"result__a\"", hrefs, texts);
                for (size_t i = 0; i < hrefs.size(); ++i) {
                    std::string href = hrefs[i];
                    const size_t uddg = href.find("uddg=");
                    if (uddg != std::string::npos) {
                        href = href.substr(uddg + 5);
                        const size_t amp = href.find('&');
                        if (amp != std::string::npos) href.resize(amp);
                        href = web_url_decode(href);
                    }
                    urls.push_back(href);
                    titles.push_back(web_html_to_text(texts[i]));
                }
                std::vector<std::string> sn_hrefs, sn_texts;
                scan_anchors(html, "class=\"result__snippet\"", sn_hrefs, sn_texts);
                for (const auto & t : sn_texts) snippets.push_back(web_html_to_text(t));
            }
            if (urls.empty()) {
                return {{"error", "web_search: no results parsed (engine layout changed or query blocked)"}};
            }
            std::string out = "Search results for: " + query + "\n";
            const int n = std::min((int) urls.size(), max_results);
            for (int i = 0; i < n; ++i) {
                out += "\n" + std::to_string(i + 1) + ". " + titles[i] + "\n   " + urls[i] + "\n";
                if (i < (int) snippets.size() && !snippets[i].empty()) out += "   " + snippets[i] + "\n";
            }
            return {{"plain_text_response", out}};
        } catch (const std::exception & e) {
            return {{"error", std::string("web_search failed: ") + e.what()}};
        }
    }
};

//
// public API
//

static std::vector<std::unique_ptr<server_tool>> build_tools() {
    std::vector<std::unique_ptr<server_tool>> tools;
    tools.push_back(std::make_unique<server_tool_read_file>());
    tools.push_back(std::make_unique<server_tool_file_glob_search>());
    tools.push_back(std::make_unique<server_tool_grep_search>());
    tools.push_back(std::make_unique<server_tool_exec_shell_command>());
    tools.push_back(std::make_unique<server_tool_write_file>());
    tools.push_back(std::make_unique<server_tool_edit_file>());
    tools.push_back(std::make_unique<server_tool_apply_diff>());
    tools.push_back(std::make_unique<server_tool_get_datetime>());
    tools.push_back(std::make_unique<server_tool_web_fetch>());   // fork addition (M6d)
    tools.push_back(std::make_unique<server_tool_web_search>());  // fork addition (M6d)
    return tools;
}

void server_tools::setup(const std::vector<std::string> & enabled_tools) {
    if (!enabled_tools.empty()) {
        std::unordered_set<std::string> enabled_set(enabled_tools.begin(), enabled_tools.end());
        auto all_tools = build_tools();

        // collect all known tool names for validation
        std::vector<std::string> known_names;
        known_names.reserve(all_tools.size());
        for (const auto & t : all_tools) {
            known_names.push_back(t->name);
        }

        // validate that every requested tool is known
        for (const auto & name : enabled_tools) {
            if (name == "all") continue;
            if (std::find(known_names.begin(), known_names.end(), name) == known_names.end()) {
                throw std::runtime_error(string_format(
                    "unknown tool \"%s\". available tools: %s",
                    name.c_str(),
                    string_join(known_names, ", ").c_str()));
            }
        }

        tools.clear();
        for (auto & t : all_tools) {
            if (enabled_set.count(t->name) > 0 || enabled_set.count("all") > 0) {
                tools.push_back(std::move(t));
            }
        }
    }

    handle_get = [this](const server_http_req &) -> server_http_res_ptr {
        auto res = std::make_unique<server_http_res>();
        try {
            json result = json::array();
            for (const auto & t : tools) {
                result.push_back(t->to_json());
            }
            res->data = safe_json_to_str(result);
        } catch (const std::exception & e) {
            SRV_ERR("got exception: %s\n", e.what());
            res->status = 500;
            res->data   = safe_json_to_str(format_error_response(e.what(), ERROR_TYPE_SERVER));
        }
        return res;
    };

    handle_post = [this](const server_http_req & req) -> server_http_res_ptr {
        auto res = std::make_unique<server_http_res>();
        try {
            json body = json::parse(req.body);
            std::string tool_name = body.at("tool").get<std::string>();
            json params = body.value("params", json::object());
            json result = invoke(tool_name, params);
            res->data   = safe_json_to_str(result);
        } catch (const json::exception & e) {
            res->status = 400;
            res->data   = safe_json_to_str(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
        } catch (const std::exception & e) {
            SRV_ERR("got exception: %s\n", e.what());
            res->status = 500;
            res->data   = safe_json_to_str(format_error_response(e.what(), ERROR_TYPE_SERVER));
        }
        return res;
    };
}

json server_tools::invoke(const std::string & name, const json & params) {
    for (auto & t : tools) {
        if (t->name == name) {
            return t->invoke(params);
        }
    }
    return {{"error", "unknown tool: " + name}};
}
