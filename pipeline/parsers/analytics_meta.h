#ifndef PIPEGEN_ANALYTICS_META_H_
#define PIPEGEN_ANALYTICS_META_H_
#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <vector>
#include "nvdsmeta.h"
#if defined(__has_include)
# if __has_include(<nvds_analytics_meta.h>)
#  include <nvds_analytics_meta.h>
#  define PIPEGEN_HAS_NATIVE_ANALYTICS 1
# endif
#endif
#if defined(PIPEGEN_REQUIRE_NATIVE_ANALYTICS) && !defined(PIPEGEN_HAS_NATIVE_ANALYTICS)
# error "Native analytics export requires the DeepStream nvds_analytics_meta.h development header"
#endif

namespace analytics_meta {
#ifdef PIPEGEN_HAS_NATIVE_ANALYTICS
// Bound list walks, including malformed/cyclic lists. Native metadata is owned
// by DeepStream; these readers never free or mutate it.
template<class T>
inline std::vector<const T*> collect(NvDsMetaList* list, NvDsMetaType type) {
    std::vector<const T*> found;
    unsigned walked = 0;
    for (; list && walked++ < 4096; list = list->next) {
        const NvDsUserMeta* meta = static_cast<const NvDsUserMeta*>(list->data);
        if (meta && meta->base_meta.meta_type == type && meta->user_meta_data)
            found.push_back(static_cast<const T*>(meta->user_meta_data));
    }
    std::stable_sort(found.begin(), found.end(), [](const T* a, const T* b) {
        return a->unique_id < b->unique_id;
    });
    return found;
}
inline std::vector<const NvDsAnalyticsFrameMeta*> frames(const NvDsFrameMeta* fm) {
    if (!fm || !fm->frame_user_meta_list) return {};
    return collect<NvDsAnalyticsFrameMeta>(fm->frame_user_meta_list,
                                          NVDS_USER_FRAME_META_NVDSANALYTICS);
}
inline std::vector<const NvDsAnalyticsObjInfo*> objects(const NvDsObjectMeta* ob) {
    if (!ob || !ob->obj_user_meta_list) return {};
    return collect<NvDsAnalyticsObjInfo>(ob->obj_user_meta_list,
                                        NVDS_USER_OBJ_META_NVDSANALYTICS);
}
inline bool has_frame(const NvDsFrameMeta* fm) { return !frames(fm).empty(); }
inline bool has_object(const NvDsObjectMeta* ob) { return !objects(ob).empty(); }

inline void quoted(std::string& out, const std::string& value) {
    // Labels originate in user configuration; repair invalid UTF-8 before JSON.
    gchar* valid = g_utf8_make_valid(value.data(), (gssize)value.size());
    const std::string text(valid);
    g_free(valid);
    static const char hex[] = "0123456789abcdef";
    out += '"';
    for (unsigned char c : text) {
        if (c == '"' || c == '\\') {
            out += '\\';
            out += (char)c;
        } else if (c < 0x20) {
            out += "\\u00";
            out += hex[c >> 4];
            out += hex[c & 15];
        } else {
            out += (char)c;
        }
    }
    out += '"';
}
inline std::string number(bool value) { return value ? "true" : "false"; }
template<class T> inline std::string number(T value) { return std::to_string(value); }
inline std::string key(const std::string& value) { return value; }
inline std::string key(int value) { return std::to_string(value); }
template<class Map> inline void map_json(std::string& out, const Map& values) {
    std::map<typename Map::key_type, typename Map::mapped_type> ordered(values.begin(), values.end());
    out += '{';
    bool first = true;
    for (const auto& item : ordered) {
        if (!first) out += ", ";
        first = false;
        quoted(out, key(item.first));
        out += ": ";
        out += number(item.second);
    }
    out += '}';
}
inline void strings_json(std::string& out, const std::vector<std::string>& values) {
    out += '[';
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) out += ", ";
        quoted(out, values[i]);
    }
    out += ']';
}
inline void append_frame(std::string& out, const NvDsFrameMeta* fm) {
    const auto records = frames(fm);
    if (records.empty()) return;
    out += ", \"analytics\": [";
    for (size_t i = 0; i < records.size(); ++i) {
        if (i) out += ", ";
        const auto& m = *records[i];
        out += "{\"unique_id\": " + std::to_string(m.unique_id);
        out += ", \"roi_counts\": ";
        map_json(out, m.objInROIcnt);
        out += ", \"overcrowding\": ";
        map_json(out, m.ocStatus);
        out += ", \"line_crossing_current\": ";
        map_json(out, m.objLCCurrCnt);
        out += ", \"line_crossing_cumulative\": ";
        map_json(out, m.objLCCumCnt);
        out += ", \"class_counts\": ";
        map_json(out, m.objCnt);
        out += '}';
    }
    out += ']';
}
inline void append_object(std::string& out, const NvDsObjectMeta* ob) {
    const auto records = objects(ob);
    if (records.empty()) return;
    out += ", \"analytics\": [";
    for (size_t i = 0; i < records.size(); ++i) {
        if (i) out += ", ";
        const auto& m = *records[i];
        out += "{\"unique_id\": " + std::to_string(m.unique_id);
        out += ", \"roi\": ";
        strings_json(out, m.roiStatus);
        out += ", \"direction\": ";
        quoted(out, m.dirStatus);
        out += ", \"line_crossing\": ";
        strings_json(out, m.lcStatus);
        out += ", \"overcrowding\": ";
        strings_json(out, m.ocStatus);
        out += ", \"status\": ";
        quoted(out, m.objStatus);
        out += '}';
    }
    out += ']';
}
// VMS currently displays scalar fields. Percent-encode label segments so a
// label containing '.' or '%' cannot collide with another namespaced key.
inline std::string segment(const std::string& text) {
    static const char hex[] = "0123456789ABCDEF";
    std::string out;
    for (unsigned char c : text) {
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
            (c >= '0' && c <= '9') || c == '_' || c == '-') out += (char)c;
        else {
            out += '%';
            out += hex[c >> 4];
            out += hex[c & 15];
        }
    }
    return out;
}
template<class Line, class T>
inline void count_field(Line& line, const std::string& name, T value) {
    // shape_json::Line has signed integers only. Preserve an exceptionally large
    // uint64 cumulative count as decimal text instead of wrapping negative.
    if ((uint64_t)value > (uint64_t)std::numeric_limits<long long>::max())
        line.field_s(name.c_str(), std::to_string(value).c_str());
    else line.field_i(name.c_str(), (long long)value);
}
template<class Line, class Map>
inline void map_fields(Line& line, const std::string& prefix, const Map& values) {
    std::map<typename Map::key_type, typename Map::mapped_type> ordered(values.begin(), values.end());
    for (const auto& item : ordered)
        count_field(line, prefix + segment(key(item.first)), item.second);
}
template<class Line> inline void forward_frame(Line& line, const NvDsFrameMeta* fm) {
    for (const auto* m : frames(fm)) {
        const std::string p = "analytics." + std::to_string(m->unique_id) + ".";
        map_fields(line, p + "roi_counts.", m->objInROIcnt);
        map_fields(line, p + "overcrowding.", m->ocStatus);
        map_fields(line, p + "line_crossing_current.", m->objLCCurrCnt);
        map_fields(line, p + "line_crossing_cumulative.", m->objLCCumCnt);
        map_fields(line, p + "class_counts.", m->objCnt);
    }
}
template<class Line>
inline void forward_object(Line& line, const NvDsObjectMeta* ob, size_t ordinal) {
    if (!ob) return;
    const std::string identity = ob->object_id == UNTRACKED_OBJECT_ID
        ? "untracked." + std::to_string(ordinal) : std::to_string(ob->object_id);
    for (const auto* m : objects(ob)) {
        const std::string p = "analytics." + std::to_string(m->unique_id) + ".object." + identity + ".";
        std::string roi, lc, oc;
        strings_json(roi, m->roiStatus);
        strings_json(lc, m->lcStatus);
        strings_json(oc, m->ocStatus);
        line.field_s((p + "roi").c_str(), roi.c_str());
        line.field_s((p + "direction").c_str(), m->dirStatus.c_str());
        line.field_s((p + "line_crossing").c_str(), lc.c_str());
        line.field_s((p + "overcrowding").c_str(), oc.c_str());
        line.field_s((p + "status").c_str(), m->objStatus.c_str());
    }
}
#else
inline bool has_frame(const NvDsFrameMeta*) { return false; }
inline bool has_object(const NvDsObjectMeta*) { return false; }
inline void append_frame(std::string&, const NvDsFrameMeta*) {}
inline void append_object(std::string&, const NvDsObjectMeta*) {}
template<class Line> inline void forward_frame(Line&, const NvDsFrameMeta*) {}
template<class Line> inline void forward_object(Line&, const NvDsObjectMeta*, size_t) {}
#endif
} // namespace analytics_meta
#endif
